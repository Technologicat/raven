# file_dialog 3.1
# MIT licensed

__all__ = ["FileDialog"]

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import glob
import os
import platform
import psutil
import threading
import time

import dearpygui.dearpygui as dpg

from ... import animation


# Hotkey support
visible_dialog_instance = None  # fdialog is modal so There Can Be Only One (TM). If needed, could use a list, and check which one has keyboard focus, but that might not always work.
def fdialog_hotkeys_callback(sender, app_data):
    if visible_dialog_instance is None:
        return

    key = app_data  # for documentation only
    # shift_pressed = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
    ctrl_pressed = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)

    # TODO: Add hotkeys to navigate up/down in the table, descend into folder, ...
    if key == dpg.mvKey_Return:
        visible_dialog_instance.ok()
    elif key == dpg.mvKey_Escape:
        visible_dialog_instance.cancel()
    elif key == dpg.mvKey_F5:
        visible_dialog_instance.refresh()
    elif ctrl_pressed and key == dpg.mvKey_Home:
        visible_dialog_instance.back_to_default_path()
    elif ctrl_pressed and key == dpg.mvKey_F:
        dpg.focus_item(visible_dialog_instance.search_field)


class FileDialog:
    _class_init_lock = threading.Lock()  # thread-safe asset loading
    _class_initialized = False

    @classmethod
    def _initialize_class(cls):
        with cls._class_init_lock:
            if cls._class_initialized:
                return
            cls._class_initialized = True

            # register our hotkey handler
            with dpg.handler_registry(tag="fdialog_handler_registry"):  # global (whole viewport)
                dpg.add_key_press_handler(tag="fdialog_hotkeys_handler", callback=fdialog_hotkeys_callback)

            cls.fd_img_path = os.path.join(os.path.dirname(__file__), "images")

            # file dialog theme
            with dpg.theme() as cls.selec_alignt:
                with dpg.theme_component(dpg.mvThemeCat_Core):
                    dpg.add_theme_style(dpg.mvStyleVar_SelectableTextAlign, x=0, y=.5)

            with dpg.theme() as cls.size_alignt:
                with dpg.theme_component(dpg.mvThemeCat_Core):
                    dpg.add_theme_style(dpg.mvStyleVar_SelectableTextAlign, x=1, y=.5)

            # texture loading
            diwidth, diheight, _, didata = dpg.load_image(os.path.join(cls.fd_img_path, "document.png"))
            hwidth, hheight, _, hdata = dpg.load_image(os.path.join(cls.fd_img_path, "home.png"))
            afiwidth, afiheight, _, afidata = dpg.load_image(os.path.join(cls.fd_img_path, "add_folder.png"))
            afwidth, afheight, _, afdata = dpg.load_image(os.path.join(cls.fd_img_path, "add_file.png"))
            mfwidth, mfheight, _, mfdata = dpg.load_image(os.path.join(cls.fd_img_path, "mini_folder.png"))
            fiwidth, fiheight, _, fidata = dpg.load_image(os.path.join(cls.fd_img_path, "folder.png"))
            mdwidth, mdheight, _, mddata = dpg.load_image(os.path.join(cls.fd_img_path, "mini_document.png"))
            mewidth, meheight, _, medata = dpg.load_image(os.path.join(cls.fd_img_path, "mini_error.png"))
            rwidth, rheight, _, rdata = dpg.load_image(os.path.join(cls.fd_img_path, "refresh.png"))
            hdwidth, hdheight, _, hddata = dpg.load_image(os.path.join(cls.fd_img_path, "hd.png"))
            pwidth, pheight, _, pdata = dpg.load_image(os.path.join(cls.fd_img_path, "picture.png"))
            bpwidth, bpheight, _, bpdata = dpg.load_image(os.path.join(cls.fd_img_path, "big_picture.png"))
            pfwidth, pfheight, _, pfdata = dpg.load_image(os.path.join(cls.fd_img_path, "picture_folder.png"))
            dwidth, dheight, _, ddata = dpg.load_image(os.path.join(cls.fd_img_path, "desktop.png"))
            vwidth, vheight, _, vdata = dpg.load_image(os.path.join(cls.fd_img_path, "videos.png"))
            mwidth, mheight, _, mdata = dpg.load_image(os.path.join(cls.fd_img_path, "music.png"))
            dfwidth, dfheight, _, dfdata = dpg.load_image(os.path.join(cls.fd_img_path, "downloads.png"))
            dcfwidth, dcfheight, _, dcfdata = dpg.load_image(os.path.join(cls.fd_img_path, "documents.png"))
            swidth, sheight, _, sdata = dpg.load_image(os.path.join(cls.fd_img_path, "search.png"))
            bwidth, bheight, _, bdata = dpg.load_image(os.path.join(cls.fd_img_path, "back.png"))
            cwidth, cheight, _, cdata = dpg.load_image(os.path.join(cls.fd_img_path, "c.png"))
            gwidth, gheight, _, gdata = dpg.load_image(os.path.join(cls.fd_img_path, "gears.png"))
            mnwidth, mnheight, _, mndata = dpg.load_image(os.path.join(cls.fd_img_path, "music_note.png"))
            nwidth, nheight, _, ndata = dpg.load_image(os.path.join(cls.fd_img_path, "note.png"))
            owidth, oheight, _, odata = dpg.load_image(os.path.join(cls.fd_img_path, "object.png"))
            pywidth, pyheight, _, pydata = dpg.load_image(os.path.join(cls.fd_img_path, "python.png"))
            scwidth, scheight, _, scdata = dpg.load_image(os.path.join(cls.fd_img_path, "script.png"))
            vfwidth, vfheight, _, vfdata = dpg.load_image(os.path.join(cls.fd_img_path, "video.png"))
            lwidth, lheight, _, ldata = dpg.load_image(os.path.join(cls.fd_img_path, "link.png"))
            uwidth, uheight, _, udata = dpg.load_image(os.path.join(cls.fd_img_path, "url.png"))
            vewidth, veheight, _, vedata = dpg.load_image(os.path.join(cls.fd_img_path, "vector.png"))
            zwidth, zheight, _, zdata = dpg.load_image(os.path.join(cls.fd_img_path, "zip.png"))
            awidth, aheight, _, adata = dpg.load_image(os.path.join(cls.fd_img_path, "app.png"))
            iwidth, iheight, _, idata = dpg.load_image(os.path.join(cls.fd_img_path, "iso.png"))

            cls.ico_document = [diwidth, diheight, didata]
            cls.ico_home = [hwidth, hheight, hdata]
            cls.ico_add_folder = [afiwidth, afiheight, afidata]
            cls.ico_add_file = [afwidth, afheight, afdata]
            cls.ico_mini_folder = [mfwidth, mfheight, mfdata]
            cls.ico_folder = [fiwidth, fiheight, fidata]
            cls.ico_mini_document = [mdwidth, mdheight, mddata]
            cls.ico_mini_error = [mewidth, meheight, medata]
            cls.ico_refresh = [rwidth, rheight, rdata]
            cls.ico_hard_disk = [hdwidth, hdheight, hddata]
            cls.ico_picture = [pwidth, pheight, pdata]
            cls.ico_big_picture = [bpwidth, bpheight, bpdata]
            cls.ico_picture_folder = [pfwidth, pfheight, pfdata]
            cls.ico_desktop = [dwidth, dheight, ddata]
            cls.ico_videos = [vwidth, vheight, vdata]
            cls.ico_music_folder = [mwidth, mheight, mdata]
            cls.ico_downloads = [dfwidth, dfheight, dfdata]
            cls.ico_document_folder = [dcfwidth, dcfheight, dcfdata]
            cls.ico_search = [swidth, sheight, sdata]
            cls.ico_back = [bwidth, bheight, bdata]
            cls.ico_c = [cwidth, cheight, cdata]
            cls.ico_gears = [gwidth, gheight, gdata]
            cls.ico_music_note = [mnwidth, mnheight, mndata]
            cls.ico_note = [nwidth, nheight, ndata]
            cls.ico_object = [owidth, oheight, odata]
            cls.ico_python = [pywidth, pyheight, pydata]
            cls.ico_script = [scwidth, scheight, scdata]
            cls.ico_video = [vfwidth, vfheight, vfdata]
            cls.ico_link = [lwidth, lheight, ldata]
            cls.ico_url = [uwidth, uheight, udata]
            cls.ico_vector = [vewidth, veheight, vedata]
            cls.ico_zip = [zwidth, zheight, zdata]
            cls.ico_app = [awidth, aheight, adata]
            cls.ico_iso = [iwidth, iheight, idata]

            with dpg.texture_registry():
                dpg.add_static_texture(width=cls.ico_document[0], height=cls.ico_document[1], default_value=cls.ico_document[2], tag="ico_document")
                dpg.add_static_texture(width=cls.ico_home[0], height=cls.ico_home[1], default_value=cls.ico_home[2], tag="ico_home")
                dpg.add_static_texture(width=cls.ico_add_folder[0], height=cls.ico_add_folder[1], default_value=cls.ico_add_folder[2], tag="ico_add_folder")
                dpg.add_static_texture(width=cls.ico_add_file[0], height=cls.ico_add_file[1], default_value=cls.ico_add_file[2], tag="ico_add_file")
                dpg.add_static_texture(width=cls.ico_mini_folder[0], height=cls.ico_mini_folder[1], default_value=cls.ico_mini_folder[2], tag="ico_mini_folder")
                dpg.add_static_texture(width=cls.ico_folder[0], height=cls.ico_folder[1], default_value=cls.ico_folder[2], tag="ico_folder")
                dpg.add_static_texture(width=cls.ico_mini_document[0], height=cls.ico_mini_document[1], default_value=cls.ico_mini_document[2], tag="ico_mini_document")
                dpg.add_static_texture(width=cls.ico_mini_error[0], height=cls.ico_mini_error[1], default_value=cls.ico_mini_error[2], tag="ico_mini_error")
                dpg.add_static_texture(width=cls.ico_refresh[0], height=cls.ico_refresh[1], default_value=cls.ico_refresh[2], tag="ico_refresh")
                dpg.add_static_texture(width=cls.ico_hard_disk[0], height=cls.ico_hard_disk[1], default_value=cls.ico_hard_disk[2], tag="ico_hard_disk")
                dpg.add_static_texture(width=cls.ico_picture[0], height=cls.ico_picture[1], default_value=cls.ico_picture[2], tag="ico_picture")
                dpg.add_static_texture(width=cls.ico_big_picture[0], height=cls.ico_big_picture[1], default_value=cls.ico_big_picture[2], tag="ico_big_picture")
                dpg.add_static_texture(width=cls.ico_picture_folder[0], height=cls.ico_picture_folder[1], default_value=cls.ico_picture_folder[2], tag="ico_picture_folder")
                dpg.add_static_texture(width=cls.ico_desktop[0], height=cls.ico_desktop[1], default_value=cls.ico_desktop[2], tag="ico_desktop")
                dpg.add_static_texture(width=cls.ico_videos[0], height=cls.ico_videos[1], default_value=cls.ico_videos[2], tag="ico_videos")
                dpg.add_static_texture(width=cls.ico_music_folder[0], height=cls.ico_music_folder[1], default_value=cls.ico_music_folder[2], tag="ico_music_folder")
                dpg.add_static_texture(width=cls.ico_downloads[0], height=cls.ico_downloads[1], default_value=cls.ico_downloads[2], tag="ico_downloads")
                dpg.add_static_texture(width=cls.ico_document_folder[0], height=cls.ico_document_folder[1], default_value=cls.ico_document_folder[2], tag="ico_document_folder")
                dpg.add_static_texture(width=cls.ico_search[0], height=cls.ico_search[1], default_value=cls.ico_search[2], tag="ico_search")
                dpg.add_static_texture(width=cls.ico_back[0], height=cls.ico_back[1], default_value=cls.ico_back[2], tag="ico_back")
                dpg.add_static_texture(width=cls.ico_c[0], height=cls.ico_c[1], default_value=cls.ico_c[2], tag="ico_c")
                dpg.add_static_texture(width=cls.ico_gears[0], height=cls.ico_gears[1], default_value=cls.ico_gears[2], tag="ico_gears")
                dpg.add_static_texture(width=cls.ico_music_note[0], height=cls.ico_music_note[1], default_value=cls.ico_music_note[2], tag="ico_music_note")
                dpg.add_static_texture(width=cls.ico_note[0], height=cls.ico_note[1], default_value=cls.ico_note[2], tag="ico_note")
                dpg.add_static_texture(width=cls.ico_object[0], height=cls.ico_object[1], default_value=cls.ico_object[2], tag="ico_object")
                dpg.add_static_texture(width=cls.ico_python[0], height=cls.ico_python[1], default_value=cls.ico_python[2], tag="ico_python")
                dpg.add_static_texture(width=cls.ico_script[0], height=cls.ico_script[1], default_value=cls.ico_script[2], tag="ico_script")
                dpg.add_static_texture(width=cls.ico_video[0], height=cls.ico_video[1], default_value=cls.ico_video[2], tag="ico_video")
                dpg.add_static_texture(width=cls.ico_link[0], height=cls.ico_link[1], default_value=cls.ico_link[2], tag="ico_link")
                dpg.add_static_texture(width=cls.ico_url[0], height=cls.ico_url[1], default_value=cls.ico_url[2], tag="ico_url")
                dpg.add_static_texture(width=cls.ico_vector[0], height=cls.ico_vector[1], default_value=cls.ico_vector[2], tag="ico_vector")
                dpg.add_static_texture(width=cls.ico_zip[0], height=cls.ico_zip[1], default_value=cls.ico_zip[2], tag="ico_zip")
                dpg.add_static_texture(width=cls.ico_app[0], height=cls.ico_app[1], default_value=cls.ico_app[2], tag="ico_app")
                dpg.add_static_texture(width=cls.ico_iso[0], height=cls.ico_iso[1], default_value=cls.ico_iso[2], tag="ico_iso")

                cls.img_document = "ico_document"
                cls.img_home = "ico_home"
                cls.img_add_folder = "ico_add_folder"
                cls.img_add_file = "ico_add_file"
                cls.img_mini_folder = "ico_mini_folder"
                cls.img_folder = "ico_folder"
                cls.img_mini_document = "ico_mini_document"
                cls.img_mini_error = "ico_mini_error"
                cls.img_refresh = "ico_refresh"
                cls.img_hard_disk = "ico_hard_disk"
                cls.img_picture = "ico_picture"
                cls.img_big_picture = "ico_big_picture"
                cls.img_picture_folder = "ico_picture_folder"
                cls.img_desktop = "ico_desktop"
                cls.img_videos = "ico_videos"
                cls.img_music_folder = "ico_music_folder"
                cls.img_downloads = "ico_downloads"
                cls.img_document_folder = "ico_document_folder"
                cls.img_search = "ico_search"
                cls.img_back = "ico_back"
                cls.img_c = "ico_c"
                cls.img_gears = "ico_gears"
                cls.img_music_note = "ico_music_note"
                cls.img_note = "ico_note"
                cls.img_object = "ico_object"
                cls.img_python = "ico_python"
                cls.img_script = "ico_script"
                cls.img_video = "ico_video"
                cls.img_link = "ico_link"
                cls.img_url = "ico_url"
                cls.img_vector = "ico_vector"
                cls.img_zip = "ico_zip"
                cls.img_app = "ico_app"
                cls.img_iso = "ico_iso"

    def __init__(
        self,
        title="File dialog",
        tag="file_dialog",
        width=1150,
        height=650,
        min_size=(460, 320),
        dirs_only=False,
        save_mode=False,
        default_file_extension=None,
        default_path=os.getcwd(),
        filter_list=[".*", ".exe", ".bat", ".sh", ".msi", ".apk", ".bin", ".cmd", ".com", ".jar", ".out", ".py", ".pyl", ".phs", ".js", ".json", ".java", ".c", ".cpp", ".cs", ".h", ".rs", ".vbs", ".php", ".pl", ".rb", ".go", ".swift", ".ts", ".asm", ".lua", ".sh", ".bat", ".r", ".dart", ".ps1", ".html", ".htm", ".xml", ".css", ".ini", ".yaml", ".yml", ".config", ".md", ".rst", ".txt", ".rtf", ".doc", ".docx", ".pdf", ".odt", ".tex", ".log", ".csv", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".svg", ".webp", ".ico", ".psd", ".ai", ".eps", ".tga", ".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a", ".wma", ".aiff", ".mid", ".midi", ".opus", ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".mpeg", ".mpg", ".3gp", ".m4v", ".blend", ".fbx", ".obj", ".stl", ".3ds", ".dae", ".ply", ".glb", ".gltf", ".csv", ".sql", ".db", ".dbf", ".mdb", ".accdb", ".sqlite", ".xml", ".json", ".zip", ".rar", ".7z", ".tar", ".gz", ".iso", ".bz2", ".xz", ".tgz", ".cab", ".vdi", ".vmdk", ".vhd", ".vhdx", ".ova", ".ovf", ".qcow2", ".dockerfile", ".bak", ".old", ".sav", ".tmp", ".bk", ".ppack", ".mlt", ".torrent", ".ics"],
        file_filter=".*",
        callback=None,
        show_dir_size=False,
        allow_drag=True,
        multi_selection=True,
        show_shortcuts_menu=True,
        no_resize=True,
        modal=True,
        show_hidden_files=False,
        user_style=0
    ):
        """
        Arguments:
            title:                  str, File dialog window title.
            tag:                    str, File dialog window DPG tag.
            width:                  int, File dialog window width (pixels).
            height:                 int, File dialog window height (pixels).
            min_size:               (int, int), File dialog minimum size.
            dirs_only:              When True, only directories will be listed.
            save_mode:              When True, asks for a filename to save as, instead of selecting file(s) to open.
                                    In the GUI, the "Search files" field becomes the filename field. (Searching is still enabled, to help avoid accidental overwriting.)
            default_file_extension: Only used when save_mode is True. If not None, and the user specifies no file extension for the "save as" filename, this extension (e.g. ".png") is automatically added.
            default_path:           str, The default path when file_dialog starts, if it's the string 'cwd', the default path will be the current working directory.
            filter_list:            [str, ...], A list of different file extensions, for the user to choose in the file type filter. E.g. [".png", ".jpg"].
            file_filter:            str, The value of the file type filter when the dialog is opened, e.g. ".py".
            callback:               callable, When the OK or Cancel button is pressed, the file dialog will call this, sending the list of selected files. Upon cancel, the list is empty.
            show_dir_size:          If True, directories will be listed with the size of the directory and its sub-directories and files. Not recommended.
            allow_drag:             If True, the files and folders in the dialog act as a DPG drag source, so you can set up a drop target to accept them as drag'n'drops in your app. See source code for details.
            multi_selection:        If True, the user can select multiple files and folders by holding down Ctrl and clicking. If False, only one file/folder can be selected, and Ctrl does nothing.
                                    Ignored when save_mode is True.
            show_shortcuts_menu:    if True, show a child window (side panel) containing different shortcuts (like desktop and downloads), and the external and internal drives.
            no_resize:              If True, the window will not be resizable.
            modal:                  If True, use DPG modal mode; a sort of popup effect. Can cause problems if the file dialog is opened by a modal window.
            show_hidden_files:      If True, the dialog shows also hidden files and folders.
            user_style:             int, different graphical styles for file_dialog. Currently available values: 0 (full), 1 (compact).
        Returns:
            None
        """

        # args
        self.title = title
        self.tag = tag
        self.width = width
        self.height = height
        self.min_size = min_size
        self.dirs_only = dirs_only
        self.save_mode = save_mode
        self.default_file_extension = default_file_extension
        self.default_path = default_path
        self.filter_list = filter_list
        self.file_filter = file_filter
        self.callback = callback
        self.show_dir_size = show_dir_size
        self.allow_drag = allow_drag
        self.multi_selection = (not save_mode) and multi_selection
        self.show_shortcuts_menu = show_shortcuts_menu
        self.no_resize = no_resize
        self.modal = modal
        self.show_hidden_files = show_hidden_files
        self.user_style = user_style

        self.instance_tag = f"0x{id(self):x}"  # for making unique DPG tags

        self.PAYLOAD_TYPE = 'ws_' + self.tag
        self.selected_files = []
        self.shown_items = []  # for selection by search filter upon pressing ok
        self.selec_height = 16
        self.image_transparency = 100
        self.last_click_time = 0
        self.last_ok_time = 0
        self.double_click_threshold = 0.25  # seconds; adjust the time as needed.  # TODO: should really get this from OS if possible in a cross-platform way

        self._initialize_class()

        # low-level functions
        def _get_all_drives():
            all_drives = psutil.disk_partitions()

            drive_list = [drive.mountpoint for drive in all_drives if drive.mountpoint]

            if os.name == 'posix':
                for device in os.listdir('/dev'):
                    if device.startswith("sd") or device.startswith("nvme"):
                        device_path = f"/dev/{device}"
                        if device_path not in drive_list:
                            drive_list.append(device_path)

            return drive_list

        def delete_table():
            for child in dpg.get_item_children(f"explorer_{self.instance_tag}", 1):
                dpg.delete_item(child)

        def get_file_size(file_path):
            # Get the file size in bytes
            if os.path.isdir(file_path):
                if self.show_dir_size:
                    total = 0
                    for path, dirs, files in os.walk(file_path):
                        for f in files:
                            fp = os.path.join(path, f)
                            total += os.path.getsize(fp)
                    file_size_bytes = total
                else:
                    file_size_bytes = "-"
            elif os.path.isfile(file_path):
                file_size_bytes = os.path.getsize(file_path)

            # Define the units and their respective sizes
            size_units = [
                ("TB", 2**40),  # Terabyte
                ("GB", 2**30),  # Gigabyte
                ("MB", 2**20),  # Megabyte
                ("KB", 2**10),  # Kilobyte
                ("B", 1),       # Byte
            ]

            # Determine the appropriate unit for formatting
            if not file_size_bytes == "-":
                for unit, size_limit in size_units:
                    if file_size_bytes >= size_limit:
                        # Calculate the size in the selected unit
                        file_size = file_size_bytes / size_limit
                        # Return the formatted size with the unit
                        return file_size_bytes, f"{file_size:.0f} {unit}"
            else:
                return 0, "-"

            # If the file size is smaller than 1 byte or unknown
            return 0, "0 B"  # or "Unknown" or any other desired default

        def on_path_enter():
            try:
                chdir(dpg.get_value(f"ex_path_input_{self.instance_tag}"))
            except FileNotFoundError:
                message_box("Invalid path", "No such file or directory")

        def message_box(title, message):
            if not self.modal:
                with dpg.mutex():
                    viewport_width = dpg.get_viewport_client_width()
                    viewport_height = dpg.get_viewport_client_height()
                    with dpg.window(label=title, no_close=True, modal=True) as modal_id:
                        dpg.add_text(message)
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Ok", width=-1, user_data=(modal_id, True), callback=lambda: dpg.delete_item(modal_id))

                dpg.split_frame()
                width = dpg.get_item_width(modal_id)
                height = dpg.get_item_height(modal_id)
                dpg.set_item_pos(modal_id, [viewport_width // 2 - width // 2, viewport_height // 2 - height // 2])
            else:
                # TODO: We really need a message box that works while the file dialog is modal.
                logger.warning(f"message_box: Cannot display message box while file_dialog is in modal. Message follows:\n{title}:\t{message}\n")

        def open_drive(sender, app_data, user_data):
            chdir(user_data)

        def _deselect_recursive(root):
            """Deselect all selectables inside DPG widget `root`, including `root` itself."""
            if dpg.get_item_type(root) == "mvAppItemType::mvSelectable":
                dpg.set_value(root, False)
            for item in dpg.get_item_children(root, slot=1):
                _deselect_recursive(item)

        def open_file(sender, app_data, user_data):  # `user_data`: [name, fullpath, timestamp, size]
            ctrl_pressed = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)

            # Detect double-click.
            # double_clicked = dpg.is_mouse_button_double_clicked(dpg.mvMouseButton_Left)  # TODO: doesn't work, why?
            current_time = time.time()
            double_clicked = (current_time - self.last_click_time < self.double_click_threshold)
            self.last_click_time = current_time

            logger.debug(f"open_file: instance '{self.tag}' ({self.instance_tag}), sender is {sender} (tag '{dpg.get_item_alias(sender)}', type {dpg.get_item_type(sender)}, value = {dpg.get_value(sender)}), app_data = {app_data}, user_data = {user_data}, ctrl = {ctrl_pressed}, doubleclick = {double_clicked}")

            # Multi selection
            if self.multi_selection and ctrl_pressed:
                if dpg.get_value(sender) is True:
                    self.selected_files.append(user_data[1])
                elif user_data[1] in self.selected_files:
                    self.selected_files.remove(user_data[1])
            # Single selection
            else:
                dpg.set_value(sender, False)  # unselect this item  (TODO: why? double-click handling?)

                if double_clicked:
                    if user_data is not None and user_data[1] is not None:
                        if os.path.isdir(user_data[1]):
                            logger.debug(f"open_file: instance '{self.tag}' ({self.instance_tag}), Content: {dpg.get_item_label(sender)}, files: {user_data}")
                            chdir(user_data[1])
                            dpg.set_value(f"ex_search_{self.instance_tag}", "")
                        elif os.path.isfile(user_data[1]):
                            if len(self.selected_files) < 1:
                                self.selected_files.append(user_data[1])
                            self.ok()
                            return user_data[1]
                else:
                    if os.path.isfile(user_data[1]):
                        _deselect_recursive(f"explorer_{self.instance_tag}")  # unselect others
                        dpg.set_value(sender, True)  # and select this item
                        # Save mode: populate file name field from clicked file, without file extension
                        if self.save_mode:
                            basename, ext = os.path.splitext(user_data[0])
                            dpg.set_value(f"ex_search_{self.instance_tag}", basename)
                            self._update_search()
                        self.selected_files.clear()
                        self.selected_files.append(user_data[1])

        def get_directory_path(directory_name):
            try:
                # Check for Linux or MacOS
                if platform.system() in ["Linux", "Darwin"] and directory_name.lower() == "home":
                    directory_path = os.path.expanduser("~")
                # Check for Windows
                elif platform.system() == "Windows" and directory_name.lower() == "home":
                    directory_path = os.path.expanduser("~")
                else:
                    # Attempt to join the home directory with the specified directory name
                    directory_path = os.path.join(os.path.expanduser("~"), directory_name)

                # Verify if the directory exists
                os.listdir(directory_path)  # Test access
            except FileNotFoundError:
                # Search for the directory in the user's home folder
                search_path = os.path.expanduser("~/*/" + directory_name)
                directory_path = glob.glob(search_path)
                if directory_path:
                    try:
                        os.listdir(directory_path[0])  # Test access to the found path
                        directory_path = directory_path[0]  # Use the found path
                    except FileNotFoundError:
                        message_box("File dialog - Error", "Could not find the selected directory")
                        return "."
                else:
                    message_box("File dialog - Error", "Could not find the selected directory")
                    return "."

            return directory_path

        def _is_hidden(filepath):
            name = os.path.basename(os.path.abspath(filepath))
            return name.startswith('.') or (os.name == 'nt' and _has_hidden_attribute(filepath))

        def _has_hidden_attribute(filepath):
            try:
                import ctypes
                FILE_ATTRIBUTE_HIDDEN = 0x2
                attrs = ctypes.windll.kernel32.GetFileAttributesW(str(filepath))
                return FILE_ATTRIBUTE_HIDDEN & attrs
            except:  # TODO: Catch which types exactly? `Exception`? `BaseException`?
                return False

        def _makedir(item, callback, parent=f"explorer_{self.instance_tag}", size=False):
            # logger.debug(f"_makedir: instance '{self.tag}' ({self.instance_tag}), making table entry for directory '{item}' with callback {callback}")  # don't keep enabled, to avoid leaking user private data to debug log

            file_name = os.path.basename(item)
            full_path = os.path.join(os.getcwd(), file_name)
            self.shown_items.append(full_path)

            modification_time_value = os.path.getmtime(item)
            modification_time_label = time.ctime(modification_time_value)

            item_type = "Dir"
            item_size_value, item_size_label = get_file_size(item)

            kwargs_cell = {'callback': callback, 'span_columns': True, 'height': self.selec_height, 'user_data': [file_name, full_path, modification_time_value, item_size_value]}
            kwargs_file = {'tint_color': [255, 255, 255, 255]}
            with dpg.table_row(parent=parent):
                with dpg.group(horizontal=True):
                    if item_type == "Dir":
                        if _is_hidden(file_name):
                            kwargs_file = {'tint_color': [255, 255, 255, self.image_transparency], 'user_data': item_type}
                        else:

                            kwargs_file = {'tint_color': [255, 255, 255, 255], 'user_data': item_type}

                        dpg.add_image(self.img_mini_folder, **kwargs_file)
                    elif item_type == "File":
                        dpg.add_image(self.img_mini_document, **kwargs_file)

                    cell_name = dpg.add_selectable(label=file_name, **kwargs_cell)
                cell_time = dpg.add_selectable(label=modification_time_label, **kwargs_cell)
                cell_type = dpg.add_selectable(label=item_type, **kwargs_cell)
                cell_size = dpg.add_selectable(label=str(item_size_label), **kwargs_cell)

                if self.allow_drag is True:
                    drag_payload = dpg.add_drag_payload(parent=cell_name, payload_type=self.PAYLOAD_TYPE)
                dpg.bind_item_theme(cell_name, self.selec_alignt)
                dpg.bind_item_theme(cell_time, self.selec_alignt)
                dpg.bind_item_theme(cell_type, self.selec_alignt)
                dpg.bind_item_theme(cell_size, self.size_alignt)
                if self.allow_drag is True:
                    if file_name.endswith((".png", ".jpg")):
                        dpg.add_image(self.img_big_picture, parent=drag_payload)
                    elif item_type == "Dir":
                        dpg.add_image(self.img_folder, parent=drag_payload)
                    elif item_type == "File":
                        dpg.add_image(self.img_document, parent=drag_payload)

        def _makefile(item, callback, parent=f"explorer_{self.instance_tag}"):
            # logger.debug(f"_makefile: instance '{self.tag}' ({self.instance_tag}), making table entry for file '{item}' with callback {callback}")  # don't keep enabled, to avoid leaking user private data to debug log

            if self.file_filter == ".*" or item.endswith(self.file_filter):
                file_name = os.path.basename(item)
                full_path = os.path.join(os.getcwd(), file_name)
                self.shown_items.append(full_path)

                modification_time_value = os.path.getmtime(item)
                modification_time_label = time.ctime(modification_time_value)

                item_type = "File"
                item_size_value, item_size_label = get_file_size(item)

                kwargs_cell = {'callback': callback, 'span_columns': True, 'height': self.selec_height, 'user_data': [file_name, full_path, modification_time_value, item_size_value]}
                kwargs_file = {'tint_color': [255, 255, 255, self.image_transparency], 'user_data': item_type}

                with dpg.table_row(parent=parent):
                    with dpg.group(horizontal=True):

                        if item_type == "Dir":
                            dpg.add_image(self.img_mini_folder, **kwargs_file)
                        elif item_type == "File":

                            if _is_hidden(file_name):
                                kwargs_file = {'tint_color': [255, 255, 255, self.image_transparency], 'user_data': item_type}
                            else:
                                kwargs_file = {'tint_color': [255, 255, 255, 255], 'user_data': item_type}

                            if file_name.endswith((".dll", ".a", ".o", ".so", ".ko")):
                                dpg.add_image(self.img_gears, **kwargs_file)

                            elif file_name.endswith((".png", ".jpg", ".jpeg")):
                                dpg.add_image(self.img_picture, **kwargs_file)

                            elif file_name.endswith((".msi", ".exe", ".bat", ".bin", ".elf")):
                                dpg.add_image(self.img_app, **kwargs_file)

                            elif file_name.endswith(".iso"):
                                dpg.add_image(self.img_iso, **kwargs_file)

                            elif file_name.endswith((".zip", ".deb", ".rpm", ".tar.gz", ".tar", ".gz", ".lzo", ".lz4", ".7z", ".ppack")):
                                dpg.add_image(self.img_zip, **kwargs_file)

                            elif file_name.endswith((".png", ".jpg", ".jpeg")):
                                dpg.add_image(self.img_picture, **kwargs_file)

                            elif file_name.endswith((".py", ".pyo", ".pyw", ".pyi", ".pyc", ".pyz", ".pyd")):
                                dpg.add_image(self.img_python, **kwargs_file)

                            elif file_name.endswith(".c"):
                                dpg.add_image(self.img_c, **kwargs_file)
                            elif file_name.endswith((".js", ".json", ".cs", ".cpp", ".h", ".hpp", ".sh", ".pyl", ".rs", ".vbs", ".cmd")):
                                dpg.add_image(self.img_script, **kwargs_file)

                            elif file_name.endswith(".url"):
                                dpg.add_image(self.img_url, **kwargs_file)
                            elif file_name.endswith(".lnk"):
                                dpg.add_image(self.img_link, **kwargs_file)

                            elif file_name.endswith(".txt"):
                                dpg.add_image(self.img_note, **kwargs_file)
                            elif file_name.endswith((".mp3", ".ogg", ".wav")):
                                dpg.add_image(self.img_music_note, **kwargs_file)

                            elif file_name.endswith((".mp4", ".mov")):
                                dpg.add_image(self.img_video, **kwargs_file)

                            elif file_name.endswith((".obj", ".fbx", ".blend")):
                                dpg.add_image(self.img_object, **kwargs_file)

                            elif file_name.endswith(".svg"):
                                dpg.add_image(self.img_vector, **kwargs_file)

                            else:
                                dpg.add_image(self.img_mini_document, **kwargs_file)

                        cell_name = dpg.add_selectable(label=file_name, **kwargs_cell)
                    cell_time = dpg.add_selectable(label=modification_time_label, **kwargs_cell)
                    cell_type = dpg.add_selectable(label=item_type, **kwargs_cell)
                    cell_size = dpg.add_selectable(label=str(item_size_label), **kwargs_cell)

                    if self.allow_drag is True:
                        drag_payload = dpg.add_drag_payload(parent=cell_name, payload_type=self.PAYLOAD_TYPE)
                    dpg.bind_item_theme(cell_name, self.selec_alignt)
                    dpg.bind_item_theme(cell_time, self.selec_alignt)
                    dpg.bind_item_theme(cell_type, self.selec_alignt)
                    dpg.bind_item_theme(cell_size, self.size_alignt)
                    if self.allow_drag is True:
                        if file_name.endswith((".png", ".jpg")):
                            dpg.add_image(self.img_big_picture, parent=drag_payload)
                        elif item_type == "Dir":
                            dpg.add_image(self.img_folder, parent=drag_payload)
                        elif item_type == "File":
                            dpg.add_image(self.img_document, parent=drag_payload)

        def _go_up_one_level(sender, app_data, user_data):
            """GUI callback: if this item double-clicked, go up one level."""
            ctrl_pressed = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)
            current_time = time.time()
            double_clicked = (current_time - self.last_click_time < self.double_click_threshold)
            self.last_click_time = current_time

            dpg.set_value(sender, False)  # unselect the ".." entry

            if ctrl_pressed:
                return
            if double_clicked:
                dpg.set_value(f"ex_search_{self.instance_tag}", "")
                chdir("..")

        def filter_combo_selector(sender, app_data):
            filter_file = dpg.get_value(sender)
            self.file_filter = filter_file
            cwd = os.getcwd()
            reset_dir(default_path=cwd)

        def chdir(path):
            try:
                os.chdir(path)
                cwd = os.getcwd()
                reset_dir(default_path=cwd)
            except PermissionError as e:
                message_box("File dialog - PerimssionError", f"Cannot open the folder because is a system folder or the access is denied\n\nMore info:\n{e}")
            except NotADirectoryError as e:
                message_box("File dialog - not a directory", f"The selected item is not a directory, but a file.\n\nMore info:\n{e}")
        self.chdir = chdir  # needs to be accessible from the outside; uses closure data from this scope, so shouldn't be injected as an instance method (on the class); inject as a regular function *on the instance*.

        def reset_dir(file_name_filter=None, default_path=self.default_path):
            logger.debug(f"reset_dir: instance '{self.tag}' ({self.instance_tag}), called with file_name_filter = {file_name_filter}, default_path = '{str(default_path)}'")
            self.selected_files.clear()
            self.shown_items.clear()
            try:
                dpg.configure_item(f"ex_path_input_{self.instance_tag}", default_value=os.getcwd())
                _dir = os.listdir(default_path)
                delete_table()

                # Separate directories and files
                dirs = [file for file in _dir if os.path.isdir(file)]
                files = [file for file in _dir if os.path.isfile(file)]

                dirs = list(sorted(dirs))
                files = list(sorted(files))

                # 'special directory' that sends back to the previous directory
                with dpg.table_row(parent=f"explorer_{self.instance_tag}"):
                    with dpg.group(horizontal=True):
                        kwargs_file = {'tint_color': [255, 255, 255, 255], 'user_data': 'Dir'}
                        dpg.add_image(self.img_mini_folder, **kwargs_file)
                        dpg.add_selectable(label="..", callback=_go_up_one_level, span_columns=True, height=self.selec_height)

                # dir list
                for _dir in dirs:
                    if not _is_hidden(_dir) or self.show_hidden_files:
                        if file_name_filter:
                            if file_name_filter in _dir:
                                _makedir(_dir, open_file)
                        else:
                            _makedir(_dir, open_file)

                # file list
                if not self.dirs_only:
                    for file in files:
                        if (not _is_hidden(file)) or self.show_hidden_files:
                            if file_name_filter:
                                if file_name_filter in file:
                                    _makefile(file, open_file)
                            else:
                                _makefile(file, open_file)

                reapply_latest_sort()  # apply the latest sort criterion (if any) explicitly (the sort callback doesn't get called automatically when we rebuild the table)

            # exceptions
            except FileNotFoundError:
                logger.error(f"reset_dir: instance '{self.tag}' ({self.instance_tag}), invalid path: '{str(default_path)}'")
            except Exception as e:
                message_box("File dialog - Error", f"An unknown error has occured when listing the items, More info:\n{e}")
        self.reset_dir = reset_dir  # needs to be accessible from the outside; uses closure data from this scope, so shouldn't be injected as an instance method (on the class); inject as a regular function *on the instance*.

        # Adapted from the table sorting example in the DPG API docs:
        #     https://dearpygui.readthedocs.io/en/latest/documentation/tables.html#sorting
        self._sort_last_sender = None
        self._sort_last_specs = None
        def reapply_latest_sort():  # need to do this manually when we rebuild the table
            if self._sort_last_sender is not None and self._sort_last_specs is not None:
                table_sort_callback(self._sort_last_sender, self._sort_last_specs)

        def table_sort_callback(sender, sort_specs):
            # sort_specs scenarios:
            #   1. no sorting -> sort_specs == None
            #   2. single sorting -> sort_specs == [[column_id, direction]]
            #   3. multi sorting -> sort_specs == [[column_id, direction], [column_id, direction], ...]
            #
            # notes:
            #   1. direction is ascending if == 1
            #   2. direction is ascending if == -1

            self._sort_last_sender = sender
            self._sort_last_specs = sort_specs

            # no sorting case
            if sort_specs is None:
                return
            assert len(sort_specs) == 1  # multi sort not supported

            # print(column_id, dpg.get_item_label(column_id))  # this is the table column widget that was clicked  # DEBUG
            # print(sender, dpg.get_item_type(sender), dpg.get_item_alias(sender))  # the sender is the table widget itself  # DEBUG
            # print([dpg.get_item_type(x) for x in dpg.get_item_children(sender, 0)])  # table columns use slot 0  # DEBUG

            column_id, direction = sort_specs[0]
            columns = dpg.get_item_children(sender, 0)  # -> list of DPG IDs; table columns use slot 0
            rows = dpg.get_item_children(sender, 1)  # -> list of DPG IDs

            do_reverse = sort_specs[0][1] < 0
            if not do_reverse:
                item_type_to_sort_key = {"Dir": 0, "File": 1}
            else:
                item_type_to_sort_key = {"Dir": 1, "File": 0}  # make directories go first also in reversed sort order
            column_idx = columns.index(column_id)

            # Create a list that can be sorted based on a cell value, keeping track of row and value used to sort.
            sortable_list = []
            for row in rows[1:]:  # ignore the magic "..", it always goes first
                # for the magic "..": ['mvAppItemType::mvGroup']
                # for all others:     ['mvAppItemType::mvGroup', 'mvAppItemType::mvSelectable', 'mvAppItemType::mvSelectable', 'mvAppItemType::mvSelectable']
                # print([dpg.get_item_type(x) for x in dpg.get_item_children(row, 1)])  # DEBUG
                cells = dpg.get_item_children(row, 1)  # in this row

                icon_widget, filename_widget = dpg.get_item_children(cells[0], 1)  # unpack the group widget
                item_type = dpg.get_item_user_data(icon_widget)  # `reset_dir`, `_makedir`, `_makefile` stash the item type here
                if column_idx == 0:  # name
                    file_name, full_path, modification_time_value, item_size_value = dpg.get_item_user_data(filename_widget)  # see `_makedir`, `_makefile`; for column 0, this is inside the group
                else:
                    file_name, full_path, modification_time_value, item_size_value = dpg.get_item_user_data(cells[column_idx])  # for other columns, the selectable is the only widget in the table cell

                # pick the sort key
                if column_idx == 0:  # name
                    data_sort_key = file_name
                elif column_idx == 1:  # date
                    data_sort_key = modification_time_value
                elif column_idx == 2:  # type
                    data_sort_key = item_type
                elif column_idx == 3:  # size
                    data_sort_key = item_size_value

                # print(icon, dpg.get_item_type(icon), dpg.get_item_user_data(icon))  # DEBUG
                sortable_list.append([row, item_type_to_sort_key[item_type], data_sort_key])

            # We'll read off the rows after we're done.
            def _sorter(elt):
                row, item_type_sort_key, data_sort_key = elt
                return (item_type_sort_key, data_sort_key)

            sortable_list.sort(key=_sorter, reverse=do_reverse)

            # create list of just sorted row ids
            new_order = [rows[0]]  # the magic ".."
            for elt in sortable_list:
                row, item_type_sort_key, filename = elt
                new_order.append(row)

            dpg.reorder_items(sender, 1, new_order)

        # main file dialog header
        with dpg.window(label=self.title, tag=self.tag, on_close=self.cancel, no_resize=self.no_resize, show=False, modal=self.modal, width=self.width, height=self.height, min_size=self.min_size, no_collapse=True, pos=(50, 50)):
            info_px = 90

            # horizontal group (shot_menu + dir_list)
            with dpg.group(horizontal=True):
                # shortcut menu
                if (self.user_style == 0):
                    with dpg.child_window(tag=f"shortcut_menu_{self.instance_tag}", width=200, resizable_x=True, show=self.show_shortcuts_menu, height=-info_px):
                        home = get_directory_path("Home")
                        desktop = get_directory_path("Desktop")
                        downloads = get_directory_path("Downloads")
                        images = get_directory_path("Pictures")
                        documents = get_directory_path("Documents")
                        musics = get_directory_path("Music")
                        videos = get_directory_path("Videos")

                        with dpg.group(horizontal=True):
                            dpg.add_image(self.img_home)
                            dpg.add_menu_item(label="Home", callback=lambda: chdir(home))
                        with dpg.group(horizontal=True):
                            dpg.add_image(self.img_desktop)
                            dpg.add_menu_item(label="Desktop", callback=lambda: chdir(desktop))
                        with dpg.group(horizontal=True):
                            dpg.add_image(self.img_downloads)
                            dpg.add_menu_item(label="Downloads", callback=lambda: chdir(downloads))
                        with dpg.group(horizontal=True):
                            dpg.add_image(self.img_picture_folder)
                            dpg.add_menu_item(label="Images", callback=lambda: chdir(images))
                        with dpg.group(horizontal=True):
                            dpg.add_image(self.img_document_folder)
                            dpg.add_menu_item(label="Documents", callback=lambda: chdir(documents))
                        with dpg.group(horizontal=True):
                            dpg.add_image(self.img_music_folder)
                            dpg.add_menu_item(label="Music", callback=lambda: chdir(musics))
                        with dpg.group(horizontal=True):
                            dpg.add_image(self.img_videos)
                            dpg.add_menu_item(label="Videos", callback=lambda: chdir(videos))

                        dpg.add_separator()

                        # i/e drives list
                        with dpg.group():
                            drives = _get_all_drives()
                            for drive in drives:
                                with dpg.group(horizontal=True):
                                    dpg.add_image(self.img_hard_disk)
                                    dpg.add_menu_item(label=drive, user_data=drive, callback=open_drive)

                elif (self.user_style == 1):
                    with dpg.child_window(tag=f"shortcut_menu_{self.instance_tag}", width=40, show=self.show_shortcuts_menu, height=-info_px):
                        home = get_directory_path("Home")
                        desktop = get_directory_path("Desktop")
                        downloads = get_directory_path("Downloads")
                        images = get_directory_path("Pictures")
                        documents = get_directory_path("Documents")
                        musics = get_directory_path("Music")
                        videos = get_directory_path("Videos")

                        dpg.add_image_button(self.img_home, callback=lambda: chdir(home))
                        dpg.add_image_button(self.img_desktop, callback=lambda: chdir(desktop))
                        dpg.add_image_button(self.img_downloads, callback=lambda: chdir(downloads))
                        dpg.add_image_button(self.img_picture_folder, callback=lambda: chdir(images))
                        dpg.add_image_button(self.img_document_folder, callback=lambda: chdir(documents))
                        dpg.add_image_button(self.img_music_folder, callback=lambda: chdir(musics))
                        dpg.add_image_button(self.img_videos, callback=lambda: chdir(videos))

                        dpg.add_separator()

                        with dpg.group():
                            drives = _get_all_drives()
                            for drive in drives:
                                dpg.add_image_button(texture_tag=self.img_hard_disk, label=drive, user_data=drive, callback=open_drive)

                with dpg.child_window(height=-info_px):
                    # main explorer header
                    with dpg.group():
                        with dpg.group(horizontal=True):
                            self.button_refresh = dpg.add_image_button(self.img_refresh, tag=f"button_refresh_{self.instance_tag}")
                            with dpg.tooltip(self.button_refresh):
                                dpg.add_text("Refresh the current folder listing [F5]")  # TODO: move the hotkey handler for this dialog here
                            self.button_back_to_default_path = dpg.add_image_button(self.img_back, tag=f"button_back_to_default_path_{self.instance_tag}")
                            with dpg.tooltip(self.button_back_to_default_path):
                                dpg.add_text("Go back to the default path [Ctrl+Home]")  # TODO: move the hotkey handler for this dialog here
                            dpg.set_item_callback(self.button_refresh, self.refresh)
                            dpg.set_item_callback(self.button_back_to_default_path, self.back_to_default_path)

                            dpg.add_input_text(hint="Path", on_enter=True, callback=on_path_enter, default_value=os.getcwd(), width=-1, tag=f"ex_path_input_{self.instance_tag}")

                        with dpg.group(horizontal=True):
                            search_hint = "Search files [Ctrl+F]" if not save_mode else "Filename to save as [Ctrl+F]"  # TODO: move the hotkey handler for this dialog here
                            self.search_field = dpg.add_input_text(hint=search_hint, callback=self._update_search, tag=f"ex_search_{self.instance_tag}", width=-1)

                        # main explorer table header
                        with dpg.table(
                            tag=f'explorer_{self.instance_tag}',
                            height=-1,
                            width=-1,
                            resizable=True,
                            policy=dpg.mvTable_SizingStretchProp,
                            borders_innerV=True,
                            reorderable=True,
                            hideable=True,
                            sortable=True,
                            callback=table_sort_callback,
                            scrollX=True,
                            scrollY=True,
                        ):
                            iwow_name = 100
                            iwow_date = 50
                            iwow_type = 10
                            iwow_size = 10
                            dpg.add_table_column(label='Name', init_width_or_weight=iwow_name, tag=f"ex_name_{self.instance_tag}")
                            dpg.add_table_column(label='Date', init_width_or_weight=iwow_date, tag=f"ex_date_{self.instance_tag}")
                            dpg.add_table_column(label='Type', init_width_or_weight=iwow_type, tag=f"ex_type_{self.instance_tag}")
                            dpg.add_table_column(label='Size', init_width_or_weight=iwow_size, width=10, tag=f"ex_size_{self.instance_tag}")

            with dpg.group(horizontal=True):
                dpg.add_spacer(width=480)
                dpg.add_text('File type filter')
                dpg.add_combo(items=self.filter_list,
                              callback=filter_combo_selector, default_value=self.file_filter, width=-1)

            with dpg.group(horizontal=True):
                self.spacer_notification = dpg.add_spacer(width=int(self.width * 0.5))
                self.text_notification = dpg.add_text("")

            with dpg.group(horizontal=True):
                self.spacer_okcancel = dpg.add_spacer(width=int(self.width * 0.5))
                self.btn_ok = dpg.add_button(label="OK", width=100, tag=self.tag + "_return", callback=self.ok)
                self.btn_cancel = dpg.add_button(label="Cancel", width=100, callback=self.cancel)

            if self.default_path == "cwd":
                chdir(os.getcwd())
            else:
                chdir(self.default_path)

    # high-level functions
    def show_file_dialog(self):
        self.chdir(os.getcwd())
        dpg.show_item(self.tag)

        global visible_dialog_instance
        visible_dialog_instance = self

        # Align the OK/Cancel buttons to the right
        dpg.split_frame()
        old_width = dpg.get_item_width(self.spacer_okcancel)
        new_width = self.width - (dpg.get_item_width(self.btn_ok) +
                                  dpg.get_item_width(self.btn_cancel) +
                                  33)  # 33: magical constant matching the default theme, to align the buttons to the right edge of the file type picker. 3 * (8 (outer padding) + 3 (inner padding))?
        logger.debug(f"show_file_dialog: instance '{self.tag}' ({self.instance_tag}), window width = {self.width}, spacer old width = {old_width}, new width = {new_width}")
        dpg.set_item_width(self.spacer_okcancel, new_width)
        dpg.set_item_width(self.spacer_notification, new_width)

    def refresh(self):
        cwd = os.getcwd()
        logger.debug(f"refresh: instance '{self.tag}' ({self.instance_tag}), refreshing at cwd = '{cwd}'")
        self.reset_dir(default_path=cwd)
        # Raven: Acknowledge the action in the GUI.
        animation.animator.add(animation.ButtonFlash(message="",
                                                     target_button=self.button_refresh,
                                                     target_tooltip=None,
                                                     target_text=None,
                                                     original_theme=dpg.get_item_theme(self.button_refresh),
                                                     duration=1.0))

    def back_to_default_path(self):
        logger.debug(f"back_to_default_path: instance '{self.tag}' ({self.instance_tag}), going back to '{self.default_path}'")
        self.chdir(self.default_path)
        # Raven: Acknowledge the action in the GUI.
        animation.animator.add(animation.ButtonFlash(message="",
                                                     target_button=self.button_back_to_default_path,
                                                     target_tooltip=None,
                                                     target_text=None,
                                                     original_theme=dpg.get_item_theme(self.button_back_to_default_path),
                                                     duration=1.0))

    def _update_search(self):
        res = dpg.get_value(f"ex_search_{self.instance_tag}")
        self.reset_dir(default_path=os.getcwd(), file_name_filter=res)

    def ok(self):
        """Close dialog and accept currently selected files.

        The list of selected files is sent to `callback`.
        """
        if not self.selected_files:
            logger.debug(f"ok: instance '{self.tag}' ({self.instance_tag}), no file(s) selected from the GUI table; figuring out what to do.")

            if self.save_mode:
                logger.debug(f"ok: instance '{self.tag}' ({self.instance_tag}), this dialog is in save mode; using content of search field as the 'save as' filename.")
                save_as_file_name = dpg.get_value(f"ex_search_{self.instance_tag}")
                if not save_as_file_name:
                    logger.debug(f"ok: instance '{self.tag}' ({self.instance_tag}), search field is empty, cannot save with empty filename; rejecting the ok.")
                    animation.animator.add(animation.ButtonFlash(message="Please enter a filename",
                                                                 target_button=self.btn_ok,
                                                                 target_tooltip=None,
                                                                 target_text=self.text_notification,
                                                                 original_theme=dpg.get_item_theme(self.btn_ok),
                                                                 flash_color=(255, 32, 32),  # orange for warning
                                                                 text_color=(255, 255, 255),
                                                                 duration=1.0))
                    return
                full_path = os.path.join(os.getcwd(), save_as_file_name)
                self.selected_files.append(full_path)
            else:  # "open file" (or directory) mode
                logger.debug(f"ok: instance '{self.tag}' ({self.instance_tag}), this dialog is in 'open file' mode; checking if we can select all item(s) shown.")
                if len(self.shown_items) == 1:  # This allows typing into search until there is a unique match, and then pressing ok to open that item.
                    logger.debug(f"ok: instance '{self.tag}' ({self.instance_tag}), exactly one item is shown; selecting that item.")
                    self.selected_files.append(self.shown_items[0])
                elif len(self.shown_items) > 1:  # ...and the same for multiple items in `multi_selection` mode.
                    if self.multi_selection:
                        logger.debug(f"ok: instance '{self.tag}' ({self.instance_tag}), multiple items are shown, multi_selection is enabled; selecting all of them.")
                        self.selected_files.extend(self.shown_items)
                    else:
                        logger.debug(f"ok: instance '{self.tag}' ({self.instance_tag}), multiple items are shown, multi_selection is disabled; rejecting the ok.")
                        animation.animator.add(animation.ButtonFlash(message="Please select an item",
                                                                     target_button=self.btn_ok,
                                                                     target_tooltip=None,
                                                                     target_text=self.text_notification,
                                                                     original_theme=dpg.get_item_theme(self.btn_ok),
                                                                     flash_color=(255, 32, 32),  # orange for warning
                                                                     text_color=(255, 255, 255),
                                                                     duration=1.0))
                        return
                else:
                    logger.debug(f"ok: instance '{self.tag}' ({self.instance_tag}), no items shown (maybe nothing matches the search?); rejecting the ok.")
                    if self.multi_selection:
                        msg = "Please select at least one item"
                    else:
                        msg = "Please select an item"
                    animation.animator.add(animation.ButtonFlash(message=msg,
                                                                 target_button=self.btn_ok,
                                                                 target_tooltip=None,
                                                                 target_text=self.text_notification,
                                                                 original_theme=dpg.get_item_theme(self.btn_ok),
                                                                 flash_color=(255, 32, 32),  # orange for warning
                                                                 text_color=(255, 255, 255),
                                                                 duration=1.0))
                    return
        assert len(self.selected_files)  # at least one file selected if we get here

        # Save mode: Ensure presence of file extension.
        if self.save_mode and self.default_file_extension is not None:
            def ensure_ext(path):
                path_lower = path.lower()
                if not any(path_lower.endswith(ext.lower()) for ext in self.filter_list):  # any valid ext is fine, but if none match, add the default ext.
                    logger.debug(f"ok: instance '{self.tag}' ({self.instance_tag}), automatically adding default file extension '{self.default_file_extension}' to '{path}'.")
                    return path + self.default_file_extension
                return path
            new_selected_files = [ensure_ext(path) for path in self.selected_files]
            self.selected_files.clear()
            self.selected_files.extend(new_selected_files)

        # Save mode: Require another click of OK (within a short time) (or a triple-click, or two double-clicks, of the filename in the list) to confirm overwrite.
        # This is a non-intrusive UI that doesn't need another modal dialog.
        confirm_duration = 2.0
        current_time = time.time()
        double_okd = (current_time - self.last_ok_time < confirm_duration)
        self.last_ok_time = current_time
        if self.save_mode and os.path.exists(self.selected_files[0]) and not double_okd:
            # Raven: Acknowledge the action in the GUI.
            animation.animator.add(animation.ButtonFlash(message="Press again to overwrite file",
                                                         target_button=self.btn_ok,
                                                         target_tooltip=None,
                                                         target_text=self.text_notification,
                                                         original_theme=dpg.get_item_theme(self.btn_ok),
                                                         flash_color=(255, 32, 32),  # orange for warning
                                                         text_color=(255, 255, 255),
                                                         duration=confirm_duration))
            return

        logger.debug(f"ok: instance '{self.tag}' ({self.instance_tag}), hiding dialog and returning {self.selected_files}.")
        dpg.hide_item(self.tag)
        global visible_dialog_instance
        visible_dialog_instance = None
        if self.callback is not None:
            self.callback(self.selected_files)
        dpg.set_value(f"ex_search_{self.instance_tag}", "")  # clear the search when exiting
        self._update_search()  # note this clears `selected_files` because refreshing the view, so should be called *after* the callback.
        self.selected_files.clear()
        self.reset_dir(default_path=self.default_path)

    def cancel(self):
        """Close dialog without selecting any files.

        An empty list is sent to `callback`, so that your app can trigger any cleanup actions needed
        (e.g. re-enabling certain GUI elements or animations after a modal dialog exits).
        """
        logger.debug(f"cancel: instance '{self.tag}' ({self.instance_tag}), hiding dialog and returning empty list.")
        dpg.hide_item(self.tag)
        global visible_dialog_instance
        visible_dialog_instance = None
        if self.callback is not None:
            self.callback([])
        dpg.set_value(f"ex_search_{self.instance_tag}", "")  # clear the search when exiting
        self._update_search()

    def change_callback(self, callback):
        self.callback = callback
        dpg.configure_item(self.tag + "_return", callback=self.callback)
