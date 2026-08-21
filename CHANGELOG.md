# Changelog

**0.2.9** (in progress):

**Added**:

*Raven-librarian*

- **`raven.librarian.agent`, a scripting surface over the agent loop** — for driving Librarian's engine from your own Python, with your document corpus, the branching chat tree and the tool-calling all in play. `agent.turn(...)` runs one assistant turn and hands back a `TurnRecord` saying what it did: the reply, the reasoning the model emitted but did not send, how many tool rounds it took and which tools it called how often, whether the reply had retrieved material to stand on, and the prompts actually put on the wire. The one-liner form needs nothing but settings and a question; pass a `chattree.PersistentForest` and the whole conversation is kept on disk in Raven's own format.
  - Two defaults differ from the app on purpose. The network tools are **off** unless asked for, because a script's tool calls are real ones and nobody is watching. The automatic document search runs only when you supply a retriever.
  - No callbacks: what the GUI receives as fifteen events, a script gets as the returned record.
  - Attachments work without a datastore file. A script can hand the model a PDF or an image and keep the whole thing in memory, so a batch that attaches a paper per question leaves nothing on disk to clean up afterwards. Pass a `PersistentForest` when the run is worth keeping, and the attachments are kept with it.
  - A batch can tell a real reply from a failed one. A turn does not crash when the backend goes away — it writes the failure into the chat as a message, which is what you want when you are watching and a trap when you are not: overnight, a backend that stopped answering at question 12 leaves 188 replies that read like replies. The record says which ones the model actually wrote, so a second pass can re-run just those, and a retry branches beside the failed attempt rather than erasing it.
  - `agent.describe_turn(...)` builds the same record from a stored conversation, so a batch that saved its chats can be analyzed afterwards without hand-rolling a tree walk.

*Constellation-wide*

- **drag files straight in from the file manager.** Every GUI app now accepts a drop where it previously wanted the in-app file browser. What a drop means is whatever that app's open button already meant:
  - *Raven-librarian*: images and documents are attached to your next message, exactly as the attach button does — mixed drops included.
  - *Raven-visualizer*: a `.pickle` opens that dataset; `.bib` files open the importer with them already filled in as input.
  - *Raven-cherrypick*: a folder opens it.
  - *Raven-xdot-viewer*: a `.dot`, `.xdot` or `.gv` opens it.
  - *Raven-avatar-pose-editor*: an image with an alpha channel loads as the character; a `.json` loads emotion templates.
  - *Raven-avatar-settings-editor*: an image with transparency loads as the character, any other image as the backdrop, and a `.json` as animator settings. It has two image slots and a drag cannot be aimed at either — the drop only reports itself on release — so the image decides: a character is a cutout, a backdrop is a full frame.
  - Drop something an app cannot use and it says so, naming what you dropped and what would have worked, rather than doing nothing. A drop that arrives while a dialog is open is ignored, so it cannot answer a question you are in the middle of.
  - Works wherever the GUI toolkit's own windowing layer does: X11, macOS and Windows. Wayland is untested — please report if it does not work there.

**Changed**:

*Raven-librarian*

- the **Tools** mode toggle is now **Internet**, and it no longer overrides **Documents**. Each switch governs one group of tools outright — *Internet* the two that reach the network (`websearch`, `webfetch`), *Documents* the three that read your document database — so all four combinations mean something. Previously *Tools* sat above both: with it off and *Documents* on, you had switched your documents on and the AI still could not search them, and nothing about a switch named "Tools" suggested it overruled the one named after the thing it was overruling.
  - **Your setting carries over.** A stored *Tools* preference becomes the *Internet* setting on first start, which keeps the intent: the old switch governed web access too, so a user who had tools off gets the network off.
  - `get_current_time` answers to neither switch and is always available. The current time is injected into every reply regardless of both toggles, so withholding the tool would leave the AI reading a call it could not resolve.
  - In `raven-minichat`, `!tools` becomes `!internet`.

- **Librarian now opens even when the LLM backend cannot answer**, and says so instead of exiting. Previously a backend that was not running ended the app at startup with an error code, which the past chats, the cleanup dialog and the settings did not need. A row above the message box reports what is wrong, in the words that say what to do about it — nothing is answering at that address (is the server running? is the address right?), or the server is running with no model loaded (load one).
  - It clears itself. Start the server, or load a model, and the row turns green a few seconds later, names the model now loaded, and goes away. Clicking it checks immediately instead of waiting for the next check.
  - Nothing is polled while the backend is healthy — the row only exists while something is wrong, and the checking stops the moment it clears.
  - A backend that goes away *mid-session* is still reported by the reply that fails, which you can reroll. This row is about the state you start in.
  - `raven-minichat` does the same and keeps the REPL — `!history` and `!dump` work with nothing loaded — reporting the verdict on the console. Its `!reconnect` command is the terminal's version of clicking the row.
  - `raven-pdf2bib` and `raven-importer` still exit instead, because they can run for hours — and they now also stop when the backend is *running with no model loaded*, which previously started the run and failed every step. That one reads as a bug in Raven when it is not caught: the backend answers, so nothing looks wrong until every extraction comes back empty. Neither recovers from a backend that goes away mid-run; that is still a run to restart.

- **editing the system prompt no longer rewrites the one your existing chats were held under.** Previously the stored system prompt was overwritten at every app start, so a conversation you had last month silently acquired today's instructions and there was no way to see what it had actually been written against. Now the datastore keeps one system prompt per distinct text, and a chat stays rooted at the one it was held under. Changing the prompt back reuses the earlier one rather than making a third.
  - **On first start after upgrading, your existing chats appear under a second system prompt** — the text they were stored with, which differs from the current one. Nothing is lost or moved: the app opens where you left off, and the older ones are reached as below.
  - **The branch arrows now work on the system prompt message**, which is how you get between them. They behave as they do on any other message; at the top of the chat they step between system prompts instead of between replies.
  - **A system prompt can be deleted when it is not the one in use**, and this takes the chats held under it with it — which is the point, since that is the only thing those chats hang from. The one currently in use stays undeletable, as before. Deleting one leaves you where a new chat under the system prompt you land on would begin, rather than on the bare prompt with its greeting out of sight.
  - Cleanup understands this: chats under an older system prompt are not offered for deletion as unreachable.

- **"branch from here" now works on the AI's opening greeting**, where it was refused before. Branching sets where you are writing from and nothing else, so from a greeting it starts a new chat under that system prompt — which is a fair thing to want, and reachable anyway through the new-chat button. It stays refused on a system prompt message, where it would leave you writing from a point that shows you none of the conversation.

- the chat datastore is now `chat.json`, with its attachments in `chat.sidecars/` beside it. They were `data.json` and `data.images/` — the first said nothing about what was in it, and the second was named when images were the only thing you could attach, which stopped being true once documents could be. **Both are renamed on first start, together**, so there is nothing to do.
  - A `data.json` is adopted only if it actually reads as a chat datastore. The name is generic enough to belong to something else entirely, and the file is looked for beside whatever datastore path you configured — so if you have pointed Raven at a directory of your own, an unrelated `data.json` there is left alone.

- **the chat log now marks its ends when you reach them with the mouse wheel**, as it already did when you got there with the keyboard. The wheel is scrolled by the GUI toolkit itself, so nothing in Raven was watching it.

*Raven-visualizer*

- the info panel marks its ends when the wheel *arrives* at one, where previously it only did so once you were already there and turned the wheel again. A single click of the wheel onto the end used to be silent.

- the importer's two LLM steps — cluster keyword extraction and abstract summarization — no longer run as a conversation with the assistant character. Both outputs are parsed by the importer rather than read by a person, while the character card asks for Markdown, for a reported train of thought, and for conversational prose — all of which had to be undone before the result could be used. Each of the two prompts already states its own task, so what the character contributed was only the part working against it. Expect cleaner keyword lists, and summaries that start with the summary.

*Raven-cherrypick*

- **the thumbnail grid scrolls smoothly**, and flashes an arrow at the top or bottom edge as you arrive there, and again if you press or wheel further. It was the last view in the constellation that jumped. A rebuild — changing the filter, or the tile size — still repositions instantly, since gliding there would animate toward a position that is about to be corrected.
  - `SMOOTH_SCROLLING`, `SMOOTH_SCROLLING_STEP_PARAMETER` and `SCROLL_ENDS_HERE_DURATION` in `raven/cherrypick/config.py` tune or disable both.

- **"Open image folder" now shows you the pictures.** It opens in the thumbnail grid, listing a folder's images as you browse. Walk into a folder and press **Pick folder** to take the one you are looking at; clicking a folder and pressing the button still takes that one, and the line above the buttons names whichever it would be.
  - The images are there to be looked at, not picked — they are how you judge whether this is the right folder, instead of remembering what its name meant. So they are dimmed and do not respond to clicks, the answer this dialog gives being a folder.

*Raven-pdf2bib*

- the same for all eight extraction steps — authors, title, keywords, abstract and the rest — which now run without the character, on prompts that already tell the model its answer "will be sent to a computer program that cannot understand natural language". The per-step progress letters on stderr are unchanged.
  - when a step fails, the error report shows the model's thinking trace and its final answer laid out the way Librarian's export buttons lay them out, so a trace in an error report and a trace in an exported chat read the same way. The usual cause of an empty step is the model overthinking until the token budget runs out, which is what the trace shows.

*Constellation-wide*

- **the file dialog can be driven from the keyboard.** Every app that opens a file browser gets this. Arrow keys, Page Up / Page Down and Home / End move a cursor through the listing; **Enter** goes as deep as it can — descending into the directory under the cursor, accepting a file where there is nothing deeper — and **Ctrl+Enter** commits where you are, as the OK button does. A line above the buttons names the path OK would return, and updates as you move.
  - **Ctrl+Space marks the entry under the cursor** where the dialog was opened for picking several files, which is what Ctrl+click does with the mouse. Marking a folder now also updates the line naming what OK will return — Ctrl+click had never done that either.
  - **Alt+Up goes up one level**, and **Ctrl+Up** does the same one-handed: on a Nordic layout Alt sits only to the left of the space bar — the right-hand key is AltGr, a different key — so Alt+Up was the one chord here that needed both hands.
  - **Tab moves the caret between the find field and the listing**, which is what frees Left and Right. They are spent on the text caret while you are typing, so until Tab existed the thumbnail view could not be crossed: its rows hold several tiles each, and without a horizontal step every column but the first was unreachable.
  - **Ctrl+1 … Ctrl+9** pick the Nth offered file type, **Ctrl+Shift+1 … Ctrl+Shift+4** sort by Name, Date, Type or Size — pressing the same one again reverses it — and **Ctrl+T** toggles the thumbnail view. The controls they stand for say so in their tooltips.
  - **Ctrl+Shift+F hands the arrow keys to the file type list**, where Up / Down step through the offered types and Home / End go to the ends; each step applies at once, so you can watch the listing narrow as you go. Esc gives the keyboard back to the find field. The mnemonic pairs with Ctrl+F: one filters the listing by name, the other by type.
  - **Ctrl+L puts the caret in the path field**, which is for the paths that do not come from browsing: one pasted from a terminal or a message, or a short root like `/mnt` that is nowhere near where you are and in nobody's shortcuts. Esc puts back where you are and hands the keyboard to the find field.
    - **The field says what Enter will do with it** — green while it names a folder that exists, red once it cannot lead anywhere, and plain while you are on your way to one. So a path that is stale, or mistyped at the far end, shows it as you go rather than as a message box after you commit. The colors belong to the typing: once Enter has taken you there, the field goes plain again. A `~` is judged by the folder it stands for, while the field keeps showing what you typed.
    - It does not complete with Tab, and wants no completion: the find field already does that better, a fragment at a time in any order, with the cursor landing on the first match and Enter descending. Typing whole paths by hand is the thing this dialog is *not* for.
  - **Ctrl+B hands the arrow keys to the shortcuts panel** — the folder shortcuts and drives down the left side, which until now could only be clicked, leaving "start somewhere else entirely" a mouse-only operation. Up / Down, Page Up / Page Down and Home / End move a cursor through it, Enter goes there, and Esc gives the keyboard back to the find field. The cursor shows while the panel has the keys and clears when it hands them back, since a place is somewhere to go rather than something to select.
  - **F1 lists the lot**, on a card that names the dialog it belongs to. The keys for the listing itself have no button to carry a tooltip, so this is where they can be found at all. It offers only what the dialog in front of you can actually do: marking several files appears where several may be picked, thumbnails where there are files to show, and the text field is described as finding or as naming according to whether you are opening or saving. Esc closes the card and leaves the dialog as you left it.

- **the file dialog's find field says whether it found anything.** Its text turns green while something in the folder matches what you have typed and red when nothing does — the same three colors Raven-visualizer's search field uses — so a listing that has gone empty is distinguishable at a glance from a typo. Typing `..` counts as a match: the way up answers a search like any other name. A save dialog leaves the field uncolored, since there it names the file to be written, and a name nothing matches is the ordinary case.

- **the file dialog offers file types only where an app asked for them.** A dialog whose caller named no types used to list some 170 extensions — `.vhd`, `.qcow2`, `.msi` and the rest — which is a menu of formats the app has nothing to do with, and in a folder picker it filters a listing that holds no files at all. It now offers "all files" and nothing else.
  - **A folder picker offers no file types at all**, the `Show` control and the two keys that reach it being gone there. Choosing among folders is not something a type filter can narrow: it applies to files, and must, or it would hide the folders you navigate through to reach them. This also takes away the blank `Show` box that the pose editor's *Save all emotion templates* used to show. Raven-cherrypick's *Open image folder* loses it too — the pictures it lists are there to tell you it is the right folder, not to be chosen.
- **the cursor breathes.** The blue mark showing which entry the keyboard is on now pulses slowly, in the file dialog's list and in every thumbnail grid — Raven-cherrypick's included, where it is the same mark on a tile. It costs no frame rate when nothing else is happening: apps that drop to a low frame rate while you read keep doing so, and the pulse simply runs at that rate.
- **the file dialog can show hidden files and folders**, from a Hidden checkbox next to Thumbnails or with **Ctrl+H**. Whether they were shown was fixed when the app built its dialog and had no control at all, so a dotfile — or a config directory in a folder picker — was simply out of reach. The choice holds until you change it back.

- **the file dialog is resizable, and opens larger.** Drag its border when a directory warrants more rows than the default shows; every app that opens a file browser gets this. The new default is chosen so that reaching for the border should be the exception rather than the routine.
  - It will not shrink past the point where its own controls stop fitting — the sort buttons are fixed-width and cannot reflow, so below that size the Thumbnails checkbox would be clipped off the edge.
  - In the thumbnail view, the tiles reflow to fill the new width as you drag.

**Fixed**:

*Raven-librarian*

- **running `raven-librarian` and `raven-minichat` at once no longer loses one of the two sessions.** Each holds the whole chat datastore in memory and writes it back on exit, so whichever closed last silently discarded everything the other had done — including a chat you were in the middle of. The second app to start now says the datastore is already open, names it, and stops. Two Librarians did the same thing to each other, and are covered too.
  - The claim is released when the process ends, crash included, so there is no stale lock to notice or clean up.

- **the chat view now opens at the end of the conversation**, instead of part-way down it. On startup, and after jumping to a chat's continuation, the latest message could be below the fold — pressing End found it there, so nothing was missing, but the view had stopped short. The longer the conversation on screen, the further short it stopped.

- **the AI's opening greeting could be deleted, rerolled, continued and branched from**, none of which it is supposed to allow — and deleting it takes the entire chat below it. The four buttons ask one shared list whether the message is a greeting, and that list was computed lazily, so the first question consumed it and the rest were answered from what was left: nothing. Which reads as "not a greeting".

- two tooltips still described the attachment store as holding images, which stopped being the whole story in 0.2.8 when documents became attachable. The two buttons that open that folder — one on an attached image, one on an attached document — also gave it two different names, though it is one folder.

*Raven-cherrypick*

- **compare mode no longer skips an image, and no longer leaves the wrong picture cached behind it.** The cycle would advance — the overlay number changed, the grid badge lit up — while one of the images never appeared, its slot showing one of the others instead. A three-image comparison would show you two, the same two every loop.
  - Cancelling compare mode filed the frame it was parked on under the index of the image you were looking at *before* compare mode started. From then on that index displayed the wrong picture — in compare mode, in the main view, and in the status bar's dimensions — until something reloaded it. So the mix-up outlasted the compare session that caused it, and came back every time you returned to that image.
  - Which image was displayed wrongly depended on where you cancelled, which is why it seemed to come and go: cancel on the frame you started from and nothing went wrong at all. **No file was ever touched** — this was the in-memory cache of decoded images handing back the wrong one, and reopening the folder cleared it.
  - Separately, a newly loaded image announced itself from a background thread while the previous one was still being drawn, and the announcement was cleared after that drawing finished rather than before it — so an announcement arriving mid-draw was discarded. That one was real too, and fixed first; it was not the cause of the skipping.

- **the image-number indicator now follows the compare cycle.** The small number in the main view's bottom-left corner stayed on whichever image was current when you entered compare mode, so it named the wrong image for every frame of the loop while the overlay number and the grid highlight both moved. It now names the frame on screen, and goes back to the current image when you leave. The window title follows the cycle too, for the same reason — it named the image you had left behind.

- **Ctrl+Shift+C during a compare cycle no longer marks the wrong image as the winner.** Every other triage control is unavailable while comparing — the keys are ignored, the toolbar buttons and grid clicks are disabled — but this one chord slipped through, and it acted on whichever image was current *before* you started comparing rather than on anything you were looking at. Since marking moves files, that put a picture you had not chosen into `cherries/` and the rest of the compare set into `lemons/`. It is now ignored during the cycle, like the rest.
  - The intended sequence is unchanged and still works: press the digit of the frame you want, which leaves compare mode on that image, then Ctrl+Shift+C to crown it.

- **undo and redo are no longer available mid-comparison**, by button or by key. Both move files and then jump to what they moved, which left the grid pointing somewhere the cycle had not chosen. Ctrl+Z and Ctrl+Y were already ignored while comparing; Ctrl+Shift+Z and the two toolbar buttons were not. Leave compare mode and they work as before.

- **attaching a document no longer freezes the app while it is read.** Reading a large PDF takes seconds — nearly four, for an 8.5 MB paper — and it used to happen before the attachment appeared at all, with the whole GUI unresponsive meanwhile: no typing, no buttons, no hotkeys. The attachment chip now appears at once and reads its document in the background.
  - The chip says which state it is in: **pulsating** while its text is being read, **calm** once it is ready, **red** if the document turns out to hold no text. Hovering a red chip — its icon or its filename — says what went wrong.
  - **A message cannot be sent while an attachment is red, or still being read.** The send button is disabled and says why; the send key refuses with a flash. Previously a document with no readable text was reported in a dialog and then silently dropped, so the message went without it — which is the one outcome nobody wants, since you attached it for a reason. Remove the red chip (or wait) to send.
  - The scanned-PDF case is the common one here: a page of images has nothing for a text extractor to find. Run it through OCR first.

- **attaching a document no longer reads it twice.** Its text was extracted once when you picked the file, to tell you straight away if a PDF turned out to be scanned pages with no text in them, and then extracted all over again when the message was sent. For a large paper each pass is seconds — nearly four, for an 8.5 MB one — so the wait happened twice for no reason. The first result is now kept and reused.

*Raven-visualizer*

- the info panel's smooth scroll is now really stopped when the panel's content is rebuilt, rather than being told to stop by a call that tidies up after it and leaves it running. A scroll in flight kept moving the panel through the swap, over the position the rebuild had just restored.

*Raven-avatar*

- in the pose editor, keyboard shortcuts no longer fire behind a modal dialog. Every failed character-image or emotion load is reported through one, and the guard that suppresses hotkeys did not count it as a dialog — so the Enter that dismissed the error also did whatever Enter does in the editor behind it.
- in the settings editor, the same guard missed the backdrop-image browser, leaving hotkeys live while it was open. The app's four other file dialogs were already covered.

*Raven-xdot-viewer*

- **the keyboard shortcuts work again from app start.** Ctrl+O, Ctrl+F, F1, F11 and the rest were dead until you clicked somewhere: the search field counts as focused from the moment the window appears, with nobody having touched it, and every shortcut was being held back for it. Typing in the search field still keeps the plain keys to itself, which is what the check was for.

- dismissing an error dialog no longer also acts on the graph behind it. The dialog floats over the canvas, so clicking its button re-centered the view on whichever node happened to sit under the pointer. 0.2.8 fixed the keyboard half of this; the mouse half was still open, because the graph's handlers are global — they fire wherever the cursor is — and decided "is the mouse over the graph?" geometrically, which cannot tell that a dialog is covering it.

*Constellation-wide*

- **the file browser now says when it cannot open a folder.** Clicking a system directory, or one the OS will not let you read, did nothing whatsoever: the explanation was routed to a message box, and DPG will not draw one over a modal window — which every file browser in Raven is. So the dialog simply sat there, and the only account of what had happened went to a log nobody was reading. The reason now appears in red on the line above the buttons, and fades after a few seconds. Its other two reports — picking a file where a folder was expected, and a folder that cannot be listed at all — were lost the same way and arrive the same way now.

- **clicking the file browser's type filter no longer costs you the keyboard.** Once you had clicked it — to read what the options were, say — the shortcuts that put the caret back in the name field (Ctrl+F, Tab, Esc) stopped doing anything, silently, for as long as that dialog stayed open. The same was true of OK and Cancel, which normally close the dialog and so never showed it, except when confirming an overwrite.

- **the file browser's shortcuts work on a desktop that is not in English.** Linux renames these directories on disk — a Finnish desktop has `~/Kuvat`, not `~/Pictures` — and the browser looked for the English names only, so on such a system every shortcut but *Home* failed to find its directory and reported an error as the dialog opened. It now reads the directories the desktop actually defines. A place you genuinely do not have is left out of the panel rather than offered and broken.
  - Windows and macOS were never affected: they translate the name their file manager *shows* and keep the directory itself in English.

- the file browser's second-click confirmation no longer offers to "overwrite" a folder. Where an app asks for a directory to write into and you name one that already exists, it still asks for the second click, but now says only that the folder exists — what becomes of what is already in it is the app's business, not the dialog's to promise.

- the file browser's shortcut to your pictures is now labelled **Pictures**, after the folder it opens. It said "Images" while going to `~/Pictures` — which is what Linux, macOS and Windows all call that folder.

- **the file browser closes faster, and the button that opens it no longer looks dead afterwards.** Closing it rebuilt the whole file listing — twice, if you picked something — although the listing was already hidden and gets rebuilt on the next open anyway. The apps run one action at a time, so whatever you clicked next had to wait for that wasted work, which is why the attach button could ignore a click, its own click animation included. On a directory of ~1600 files that was roughly half a second per close.
  - Long listings are cheaper to display too: the browser now draws only the rows on screen, where it used to draw all of them on every frame.

---

**0.2.8** (7 August 2026):

**Added**:

*Raven-librarian*

- keyboard shortcuts no longer fire behind a modal dialog. Opening the attach-file browser, or any dialog, left the chat hotkeys live underneath it — so Enter could send a chat message while you were picking a file. (`raven-xdot-viewer` had the same gap on its error dialogs.)

- idle CPU/GPU throttle in the render loop. When the avatar is paused (auto-off after the configured idle timeout), no LLM turn is in flight, no RAG indexing is running, and there has been no recent user input, the GUI drops to ~12 fps instead of running flat-out. Same pattern as `raven-cherrypick`, `raven-xdot-viewer`, and `raven-avatar-pose-editor`.
- new INDEXING indicator (red, pulsating) shows while the RAG document database is being updated in the background — previously silent CPU/GPU work that read as "the app is broken". Includes per-document progress (`[14 / 186] | filename | elapsed 6 seconds, ETA 01:14, total 01:20`) and `Saving…` during the rebuild + datastore save tail. Mnemonic borrowed from audio/video apps: red = recording.
- the DOCS indicator (during RAG search) now reports per-phase progress — `Tokenizing query…`, `Embedding query…`, `Keyword search…`, `Semantic search…`, `Merging results…`.
- RAG: a search now interleaves cleanly with an in-flight indexing commit instead of blocking until the commit finishes. The slow per-document work (chunkify + tokenize + embed) runs outside the lock, so a query waits at most one document's worth of mutation time to start.
- RAG: closing the app while indexing is in progress no longer blocks until the full backlog is processed. The commit loop exits cleanly after the current document on cancellation, persists whatever was applied, and requeues the unprocessed remainder; on the next app start, `bootup`'s rescan re-detects the corresponding file changes via mtime.
- the indicators are now stacked vertically — INDEXING (red, RAG database update), DOCS (white, RAG search), SYSTEM (LLM prompt processing), WEB (websearch tool) — instead of overlapping at the same screen position. INDEXING and DOCS are independent and show simultaneously when an LLM query lands during a background reindex, each with its own progress label.

- **`raven-arxiv2bib`, a new command-line tool that fetches arXiv metadata for a list of identifiers and writes BibTeX.** This completes the path `raven-arxiv2id` starts, so a folder of downloaded papers becomes a searchable document database without leaving the constellation: `raven-arxiv2id -i ~/papers | raven-arxiv2bib -o papers.bib`, then `raven-burstbib`, then `raven-indexer`. Identifiers can also be given as arguments or in a file.
  - It waits arXiv's requested three seconds between requests, like Raven's other arXiv tools — a personal paper collection runs to hundreds or thousands of identifiers, which is exactly the scale at which politeness matters.
  - The version arXiv answered with is recorded, however the identifier was spelled — ask for `2410.07866v5` and the entry says v5; ask for the bare `2410.07866`, meaning "whatever is current", and the entry still says v5. So a bibliography records which revision it actually describes, and a set refreshed six months later differs visibly rather than silently. `--strip-versions` drops the suffix, for a bibliography meant for citing papers rather than tracking a collection.
  - Identifiers arXiv returns nothing for are reported at the end rather than aborting the run, so one withdrawn or mistyped entry cannot cost the several hundred that worked.

- **`raven-indexer`, a new command-line tool that builds or refreshes the document database without starting the GUI.** Indexing a corpus previously meant launching `raven-librarian` and waiting, which tied a batch job to a desktop session — and to the GUI being in a runnable state, so an unrelated frontend problem could block an indexing run that had nothing to do with it. Run with no arguments it indexes your configured documents directory; give it a directory to index that one instead, `-d` to write the index elsewhere, `-r` to descend into subdirectories, and `-q` for just the summary.
  - It reconciles rather than rebuilds: new files are added, changed files re-read, deleted files dropped, and files already indexed are left alone. Re-running it on an unchanged corpus takes a few seconds.
  - Interrupting it with Ctrl+C leaves a valid partial index rather than a corrupt one, and re-running resumes from there.
  - It ingests exactly what Librarian ingests, so the index it builds is the one the chat clients expect.

- the chat input is now **multiline**, for composing multi-paragraph messages. The send / microphone controls move below the text field.
  - **Ctrl+Enter sends, Enter inserts a newline.** Set `config.send_message_key = "enter"` to swap them. Ctrl+Enter is the default because a mis-aimed Enter costs much more than a mis-aimed Ctrl+Enter: sending half a message cannot be undone (there is no message editing yet — you would copy the sent text back out, paste, delete, resume), while an unwanted newline is one backspace.

- **keyboard scrolling for the chat log, and a view that stops fighting you while the AI writes.** Previously the mouse wheel was the only way through a long conversation, and the view scrolled itself to the end on every arriving chunk. Page Up / Page Down now page the log — from inside the message field too, since reaching for them while typing is how you look back at what you are replying to — Up / Down nudge it a few lines, and Home / End jump to the start and to the latest message. Scrolling is animated rather than instant, matching `raven-visualizer`.
  - **The view follows a reply only while you are already at the end of it.** Scroll up to re-read something and you stay where you put yourself, instead of being hauled back down by every chunk — which on a thinking model meant waiting out the whole turn. Scrolling back to the bottom resumes following, as does pressing End.
  - **A "jump to latest" pill** appears in the bottom-right corner when a reply arrives while you are reading further up. It says whether the AI is still writing or has finished — pulsating while it writes — and clicking it takes you to the end. It clears itself when you get there, so there is nothing to dismiss. It stays away entirely when you are simply paging back through an old conversation, since nothing has arrived.
  - **Running into either end of the log flashes an arrow band**, the same end-of-scroll feedback `raven-visualizer`'s info panel gives. Only for scrolling you asked for, so a reply arriving at the end never flashes.

- **attach images to your messages** when a vision-capable model (VLM) is loaded. A new paperclip button in the composer opens an image picker; picked images stack in a strip above the text field (each with its own remove button) and are sent with your next message — you can send an image with no text at all. Attached images render inline in the chat log, and the context-fill indicator now accounts for their token cost.
  - Attached images are stored locally beside the chat datastore (as sidecar files) and referenced internally, so a saved chat reloads offline and never phones home to fetch them — even if an image originally came from the web. Images larger than the configured cap (~1 MP by default) are downsampled before storage, keeping the full-resolution original as a second copy by default (`config.image_store_max_megapixels`, `store_original_image`).
  - The paperclip is disabled, with an explanatory tooltip, when the loaded model is known to be text-only. When the backend doesn't report vision capability (e.g. oobabooga), attachment is allowed and the backend decides whether it can use the image.
  - Each inline image carries its original filename as a tooltip and a small row of per-image actions: show the full-size saved copy, open the original source it came from (when a source location was recorded — disabled otherwise), and open the folder where images are stored. Failures (a moved or deleted file) flash a brief message on the button rather than interrupting with a dialog.

- **attach documents (plain text, PDF, office formats, or saved web pages) to your messages** — the composer's attach button takes documents on *any* model (a document is fed to the model as text, so no vision model is required, unlike an attached image). Picked documents show as chips in the strip above the text field; on send, each document's text is folded into your message, and the context-fill indicator counts it.
  - Documents are stored locally beside the chat datastore (as sidecar files) and referenced internally, exactly like attached images — a saved chat reloads offline, and the chat JSON stays small even for a large PDF. Born-digital PDFs have their text layer extracted; a scanned/image-only PDF (no text layer) is rejected at attach time with a note to OCR it first.
  - Each attached document renders inline in the chat log as a chip (document icon + filename) with the same per-item actions as images: show the saved copy (opens it in your default app), open the original source (when recorded), and open the folder where attachments are stored.

- **a web page the AI fetches now becomes an attachment too, instead of being pasted into the chat log whole.** Asking it to read a paper used to drop the entire article into the conversation — dozens of screens to scroll past on the way to the answer. A long fetch now shows its opening paragraphs and a chip, with the same actions as a document you attached yourself: open the saved copy, open the page it came from, open the attachments folder. The AI still reads every word; only the log changes.
  - The fetched text is saved beside the chat under a content hash, so a page that later goes away is still readable in the conversation that used it, and re-fetching an unchanged page costs no extra disk. It also means a huge page is now sized against the context window rather than being sent whole. A fetched page is capped at `config.docs_fetch_max_fraction_of_context` (a tenth of the window) — the same ceiling the AI's document-database reads get, because both are the AI reaching for something on a hunch, and neither should be able to crowd out the conversation it was meant to inform. A document *you* attach has no such cap: that one is an instruction to read it.
  - Short fetches stay inline, where reading them in place beats clicking a chip; the threshold is `config.tool_result_attachment_threshold` (4000 characters), and `config.tool_result_preview_characters` sets how much of the opening is shown. Web *searches* are never turned into attachments at any length — the result is a list of links, and the links are the point.

- a document the AI pulls from **your own document database** now carries the same handles as one it fetched from the web: a named chip with buttons to open the file and to reveal the documents folder. Previously a knowledge-base fetch was plain text with no way to get at the source. The document is *not* copied anywhere — it is already your file, so the buttons open the original in place.

- **attachments now open when you click them**, not only from the button underneath. Applies to inline images, attached documents, fetched pages and knowledge-base documents alike; the action row stays, since it is what distinguishes opening the saved copy from opening the original source.

- two folder-opening buttons under the avatar panel — **Open documents folder** (the RAG drop folder where you put files for the AI to search; created if it doesn't exist yet) and **Open chat data folder** (the chat history and attached images) — for quick access to both without digging through the config for their paths.

- new **Clean up & save** button, on a Maintenance row under the same panel. Deleting a chat branch frees its messages but leaves the images and documents that were attached to them on disk — attachments are shared between branches, so no single deletion can safely decide a file's fate. This reclaims them, and saves the chat data as it goes.
  - It shows what it would delete *before* deleting anything: a count of unreachable messages, and the orphaned attachments — images as a grid of thumbnails, documents as a named list with sizes. Both sections start collapsed, so a large cleanup still opens as a one-line summary.
  - Because attachments are stored under a content hash, a stray file has no readable name of its own. Each one now carries a small description written beside it when it was stored — original filename, source, timestamp — so the preview can name it, sort it, and show where it came from, long after the message that referenced it is gone. Attachments stored before this release have no description and appear under their hash.
  - Every listed item can be opened in your default application before you decide — a thumbnail is enough to remember an image by, not always enough to judge it, and a document has no thumbnail at all. A downsampled image and the full-resolution original kept beside it count as one attachment, as they do in the chat log: one entry, showing the disk both occupy, and it is the original that opens.
  - Anything you want to keep can be copied out first, per item or all at once, to a staging folder (`config.attachment_staging_dir`), under its real filename rather than its hash. One file comes out per attachment — the full-resolution original where one was kept, otherwise the stored copy; the downsample is not worth keeping when the original is right there. Copies, so cancelling the cleanup afterwards costs nothing.

- **navigation links between a tool call and its result.** When the AI uses a tool, the call appears in its message as a gear row; each now carries a down-arrow that jumps to that call's result, and each result carries an up-arrow back to the call it answers. The view scrolls there and briefly flashes the destination, so you can see where you landed. This matters most when one turn makes several calls — each arrow goes to *its own* result, and each return flashes *its own* gear, instead of leaving you to match them up by reading.
  - If a call has no recorded result — still running, interrupted, or answered on a different branch of the chat — the button says so instead of jumping.

- RAG: the **document database now ingests PDFs, office documents and saved web pages** alongside the plain-text formats — drop the file into the documents folder and its text is extracted and indexed like any other document. The ingested file types are configurable via the new `config.llm_docs_exts`.
  - Born-digital PDFs (`.pdf`), Word (`.docx`), PowerPoint (`.pptx`), the OpenDocument counterparts (`.odt`, `.odp`), and HTML (`.html`, `.htm`). The same set applies to documents attached to a chat message — one extractor serves both, so the two never accept different things. The legacy binary office formats (`.doc`, `.ppt`) are not supported, as reading them would mean calling out to a separate converter program.
  - Tables are read in place rather than appended at the end, so a value stays next to its label. On slides, text inside grouped shapes is read too, and so are presenter notes — on a lecture deck the notes are frequently where the argument lives, while the slide only states the claim.
  - A saved web page is stripped of navigation, sidebars and footers, leaving the article — the same readability extraction the `webfetch` tool applies to live pages, so a page read from disk comes out like the same page fetched. Its `<title>` is kept as a heading, since a filename saved off the web frequently names nothing.
  - Only the *text layer* is read, in every format. A document whose content is all pictures — a scanned PDF, a deck of diagrams — extracts as empty and is skipped. For a scanned PDF, running it through OCR first (e.g. `ocrmypdf --force-ocr in.pdf out.pdf`) gives it a text layer to index. A web page that builds its content with JavaScript likewise reads as empty: nothing here runs a page's scripts, because putting a file in the documents folder must never be enough to make it execute.

- **a ceiling on how many rounds of tool calls one reply may take** (`config.max_tool_call_rounds`, 20), so that a model rephrasing a search that keeps finding nothing cannot go around forever. It is a backstop rather than a working limit — a reply that finds what it needs stops well below it, and the number is set from where models actually stop: asked about something its document database had nothing on, Qwen3.6-35B-A3B rephrased the search nine or ten times before giving up and reporting the absence. A reply that does run long can be ended with Ctrl+G.
  - At the ceiling the AI is told its budget is spent, and any further call is answered with an error saying so rather than the tool being taken away. Withdrawing a tool mid-reply invalidates the language-model backend's cache of the conversation so far, and the rest of it then has to be reprocessed before the reply can continue. If the AI asks anyway, the tools *are* withdrawn, since nothing else can guarantee the reply ends; `config.max_tool_call_refusal_rounds` is how many times to try the gentler route first.

- new `webfetch` tool — the AI can retrieve a web page's main content as clean text/markdown, the natural companion to `websearch` (search → pick a link → fetch it). Static extraction with a headless-browser fallback for JS-rendered pages, and cleaner-source routing for arXiv, Reddit, and YouTube (transcript). An optional domain allowlist (off by default) constrains which sites the AI may visit on its own initiative, while URLs the user types into the chat are always honored for that turn; requests to private-network addresses and non-HTTP(S) schemes are refused server-side, and fetched text is stripped of invisible prompt-injection characters before the model sees it.
  - Every URL in the chat history now has a small inline "send to chat input" icon; clicking it drops the URL into your message draft (so you can add context and send) — the one-click way to hand a search result back to the AI as a user-typed, auto-allowed URL.
  - When the allowlist refuses a fetch, the denied result carries an "approve this host & retry" button: it allows the host for the rest of the session and re-runs that one fetch on a new conversation branch, so you don't have to edit the config and restart. Only the denied fetch re-runs — any companion `websearch` from the same step is preserved exactly, not re-queried.

- now works with **LM Studio** and other OpenAI-compatible LLM backends, not just oobabooga. The backend is autodetected at connection time (override with the new `llm_backend_flavor` setting), and the wire-format differences are handled transparently: incremental tool-call streaming (LM Studio streams a tool call in fragments where ooba sends it whole), the `[DONE]` end-of-stream sentinel, and errors delivered as an HTTP-200 SSE event. New settings in `raven.librarian.config` (all autodetect by default): `llm_backend_flavor`, `llm_model` (names the model to send per request — e.g. drives LM Studio's just-in-time loading), and `llm_tokenizer_path` (an optional local tokenizer for exact token counts on backends without a count endpoint).
  - The character card now tells the model its **actual** loaded model identity and context-window size (read from the backend), instead of fixed values.
  - New **context-fill indicator** in the bottom toolbar — current conversation size vs the loaded context window, as a percentage. It starts from a fast estimate and, after a few seconds of inactivity, an idle background "prefill" reads back the backend's exact prompt-token count *and* warms the backend's KV cache, so your next turn starts generating sooner. Tunable via `config.context_prefill_idle_delay` (`None` disables it).
  - LLM output is **no longer capped at a fixed token count by default** — generation runs until the model stops on its own, or the context window fills. Modern models reliably signal completion, and the Stop button is there for interactive use. Set `llm_sampler_config["max_tokens"]` in `raven.librarian.config` to an integer to re-impose a cap.
  - The model's **thinking trace now always shows** as a collapsible thought bubble, regardless of how the backend delivers it. Previously, only models that emit `<think>` tags inline (e.g. via oobabooga) had their reasoning displayed; backends that stream reasoning on a separate channel (LM Studio, llama.cpp) left the thinking invisible — the chat looked frozen during the whole thinking phase. Reasoning is now stored separately from the answer text.
  - **Tool-call invocations are now visible in the chat history** — the function name and arguments the AI called (e.g. `websearch(query='...')`), shown with a cogs icon between the AI's message and the tool result, instead of silently vanishing. Works whether the call arrived as a structured field or inline tags.
  - Existing chats load seamlessly under the new format: thinking traces and tool-call invocations stored inline in older chats are migrated automatically on first load (and the migration is safe to re-run).

- **AI content is now disclosed as such**, on screen and in anything you copy out (EU AI Act Article 50).
  - The notice below the chat states outright that you are interacting with an AI system, ahead of the existing note that its answers need checking. The old wording only implied it. Always visible and not dismissable.
  - Exported chat text carries **origin metadata**, as a YAML front-matter block that names the generator, the export time, which messages came from a human, which from the AI, and which model produced each AI message. Both export routes emit it: the whole-chatlog copy (F8) gets one manifest for the document, and a single copied AI or tool message gets a one-message manifest of its own, since a lifted fragment travels without the document's. Copying one of your *own* messages is unchanged — there is no AI generation to disclose, and it keeps the copy clean for editing and resending.
  - This is what a system on this side of the model boundary can honestly attest to. The robust mark for AI-generated text is a watermark applied while the model samples; *Raven-librarian* runs third-party models through an OpenAI-compatible backend and never sees the sampler, so it records the origin metadata it does know rather than claiming a mark it cannot make.

*Raven-visualizer*

- logs whether *Raven-server* is reachable once at startup, so its presence or absence is explicit from the first line of output rather than only surfacing when the importer first reaches for it. The server is optional for the Visualizer, so both outcomes log at info level.

*Raven-server*

- HTTP API: new `/api/embeddings/info` endpoint returns the loaded embedding models keyed by role (HF repo name and output vector dimension). Parallel to the existing `/api/stt/info` and `/api/tts/info`; lets clients size storage and avoid hardcoding values that drift when the server config changes.

- NVRTC sanity check at startup. Compiles a trivial element-wise kernel via the jiterator path right after device validation, so a broken NVRTC runtime (missing `libnvrtc-builtins.so`, version skew between bundled and host CUDA) surfaces as a clear startup warning instead of an opaque crash the first time a JIT-compiled path runs. Adds ~300 ms to startup on healthy CUDA setups, nothing on CPU-only ones.

*Raven-avatar*

- settings editor: `Clear` and `Default` buttons on the postprocessor section header. `Clear` disables every filter for a blank-slate starting point; `Default` reloads the postprocessor chain from `animator.json`. Per-filter `Reset` buttons now have tooltips.
- settings editor: idle CPU/GPU throttle in the render loop. When the avatar is paused (Ctrl+P) and there has been no recent user input, the GUI drops to ~12 fps; live playback runs at full fps as before. Brings the settings editor in line with the rest of the constellation.

*Raven-cherrypick*

- the window title bar now shows the current image's filename alongside the folder (`… — folder — filename`), so the open image is identifiable from the taskbar / window switcher. In fullscreen, where the window manager hides the title bar, the filename moves into the status bar instead — so long autonamed filenames don't crowd the rest of the status bar during normal windowed use.
- the cherry / lemon triage marker now sits just outside the top-right corner of the *image* rather than the corner of the viewer pane, so it stays beside the image when the image is small in a large window. It clamps back to the pane corner when the image fills the view.
- the current image's position number now shows at the lower-left of the main view (e.g. `13`), mirroring the thumbnail grid's tile numbers and the `[13 / 133]` status readout. Anchored to the image like the triage marker, so it stays put as you browse same-size images.
- **undo / redo for triage moves** — `Ctrl+Z` / `Ctrl+Shift+Z` (also `Ctrl+Y`), plus toolbar buttons. Reverts the last cherry / lemon / clear / winner action, including a multi-select batch as one step, and keeps the view on the changed image so you see what changed — staying put when you're already on it (e.g. reverting a winner+losers set leaves the winner current), only moving when needed. Works from a filtered view too. Session-only: opening a folder rescans from disk, which is the source of truth.
- **WASD navigation** as an alias for the arrow keys, plus `Q` / `E` for page up / down — so triage can be done one-handed (left hand on WASD, with the `X` / `C` / `V` triage cluster right below it) on a coffee break. Mirrors the arrows everywhere they work, including panning the focused image pane. The arrow keys keep working unchanged.

*Raven-arxiv-download*

- prints the paper's citation (`Authors (Year) - Title`) just before downloading its PDF, so it stays on screen during the rate-limit wait — a mistyped ID that resolved to the wrong paper is caught before the download completes. Only shown when a paper is actually being fetched; already-present papers are reported by their existing one-line status.
- ends with a summary counted **by outcome** — downloaded, already present, duplicate identifier, no PDF available, failed. A rerun over the same list does almost nothing, that being the point of skipping papers already present, so a bare total answers nothing; and only the outcomes that occurred are named, so a clean run does not print `0 failed` for you to read past.
- fetches metadata for up to 100 papers per request instead of one, roughly halving the wall time of a large download. arXiv's three-second politeness delay is charged per *request*, and each paper needs two of them — one for metadata, one for the PDF — so the metadata half of that cost now amortizes away: 170 papers went from 340 waits to 172. The PDF fetches are unchanged, one per paper, which is the floor.
  - A request that fails now costs its batch rather than the run, and transport failures (dropped connections, read timeouts) are retried with backoff before it comes to that. Batching makes this matter: one blip used to lose a single paper, and would otherwise now lose a hundred.
  - Papers are matched to the metadata that comes back by identifier, and a request naming a version is matched on *that* version. Asking for two versions of one paper in a single run therefore gets each its own metadata, rather than both silently receiving whichever arXiv answered with first.
- new `-s` / `--save-bib file.bib` option writes the papers' metadata as BibTeX alongside the PDFs, so downloading a set of papers and building its bibliography is now one command instead of two. This is free — naming the PDFs already requires the metadata, so nothing extra is fetched and no extra rate-limit waiting is incurred, unlike running `raven-arxiv2bib` over the same identifiers afterwards. Version suffixes are kept, since a download names a specific version and the bibliography should record which one it describes; papers already present in the output directory are included too, the bibliography being a description of the set you asked for.

*Raven-arxiv2id*

- new `-s` / `--strip-versions` option prints each identifier without its version suffix. This is what refreshes a collection of preprints: an arXiv identifier carrying a version means *that* version, and one without means whatever is current, so dropping the suffix turns the tool's output into a request for the latest of everything — `raven-arxiv2id -i ~/papers --strip-versions` piped to `raven-arxiv-download` fetches the papers that have been revised since you saved them, and to `raven-arxiv2bib` brings the bibliography along. Previously this needed hand-editing the identifier list, and there was no way to notice which papers had moved on.

*Raven-fixbib*

- **new tool**: repairs BibTeX records whose braces a parser refuses, which is how mathematics arrives when a `.bib` was built from PDFs — set-builder notation like `{0 <= rho <= 1` with its closing brace lost somewhere in the extraction. One such brace ends a field value early or leaves it unterminated, and the whole record goes missing from anything that reads the file: title, authors and all. `raven-fixbib myrefs.bib` escapes the stray braces and writes `myrefs_fixed.bib`; `--in-place` edits the original, and `--dry-run` only reports. Your bibliography is yours, so nothing is written back unless you ask.
  - Only the offending braces are escaped — the text is otherwise identical, character for character. What cannot be repaired is reported with its line number and the fields that look responsible, rather than guessed at: where a record lost a value's *terminator* rather than gaining a stray brace, nothing can know where the missing one belonged.
  - This is the other half of a fix that landed earlier for `raven-wos2bib`, which stopped *generating* unbalanced braces. This one repairs the files you already have.

*Constellation-wide*

- New `"gpu"` device string in config files: an explicit autodetect token that picks whichever GPU backend (CUDA / MPS / XPU / Vulkan) is available, falling back to CPU if none. Replaces the implicit autodetect that was meant to live inside the `"cuda"` string with a clearly-named alias — `"cuda"` now means exactly CUDA. The defaults in the server (`raven/server/config.py`, `config_avatar_only.py`, `config_lowvram.py`) and the Visualizer / Librarian client configs now use `"gpu"`. Explicit names like `"cuda:0"` or `"mps"` are still honored as deliberate choices — no cross-backend fallback. On a machine with multiple distinct GPU backends active simultaneously (rare — e.g. NVIDIA + Intel Arc), startup raises `RuntimeError` and asks for an explicit pick.

- New `--log <path>` and `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}` CLI options on all major apps (`raven-visualizer`, `raven-importer`, `raven-librarian`, `raven-server`, `raven-minichat`, `raven-xdot-viewer`, `raven-cherrypick`, `raven-conference-timer`, `raven-avatar-pose-editor`, `raven-avatar-settings-editor`) plus the bibliography tools that emit log records (`raven-pdf2bib`, `raven-wos2bib`, `raven-csv2bib`). `--log` mirrors stderr to a file (overwritten each run) so users can capture session logs for bug reports without redirecting their terminal — especially useful for the GUI apps where the launching terminal is often a side window. The logfile path accepts `~` and is resolved to an absolute path. The mirroring survives third-party libraries (notably `flair`) that call `logging.shutdown()` on import.

**Changed**:

*Raven-librarian*

- **the attach dialog stops offering image formats when the loaded model cannot read them.** On a model the backend confirms is text-only, the picker offers *Documents* and *All files* — where before it offered images too and then refused them after you had picked one. A model whose capability the backend does not report is still offered everything, as it was: unknown is not the same as no.
  - What is offered is decided when the dialog opens, so loading a vision model mid-session makes images pickable without a restart.
  - Picking an image through *All files*, or dragging one in, is still refused with the same message — the picker steers, it does not enforce, and a drop never went through the picker at all.

- the F1 help card now describes the app as it is. It named `websearch` as though it were the only tool (there are five), said the document database takes `.txt` files (it takes PDF, Word, PowerPoint, OpenDocument, HTML and the plain-text formats), and twice called Raven-librarian a tech demo. Attachments still are not covered — the card is a single fixed-size screen, and there is no room left until that changes.
- search results now have a **maximum length** (`docs_max_result_length`, default 2000 characters). A result is stitched together from however many neighbouring pieces of a document the search happened to find, and that had no upper bound — so the time the language model spent reading before it started replying varied by an order of magnitude between turns, with nothing to cap it. Long results are now returned as several results covering the same text, which bounds the wait at roughly five seconds without dropping anything. Set it to `None` for the old behavior.
- the document database now returns **50** matches per search instead of 20, which finds the right document markedly more often. Measured on a 12000-abstract corpus, the answer is present among the results 84.8% of the time at 50 against 74.7% at 20. The cost is the language model reading more text before it starts replying — about 1.7 seconds more per turn on a 30B model — and 50 is where that trade stops being worth it: going on to 100 buys half as much recall for four times the extra wait. Configurable as `docs_num_results` in `raven/librarian/config.py`; lower it if your backend is slow at reading prompts.
- **the Speculation toggle is gone, and Documents now governs everything to do with your document database** — the automatic search, the AI's own document tools, the reminder to stick to what was retrieved, and the marker below a reply. One switch, and the tools are not offered at all when it is off, so the AI cannot reach around it.
  - Speculation existed to gate a behavior that is itself gone. With it off, a question your document database had nothing on was answered by *"No matches in document database. Please try another query."* — the language model was bypassed entirely. That could not tell a question *about your documents* from a passing general question, so it refused both, and the toggle had to be flipped every time the discussion wandered.
  - **A question your documents have nothing on is now answered anyway**, with a `[no sources retrieved]` marker below the reply. The marker reports what was *retrieved*, not whether the answer used it.
  - With **Documents off** the marker does not appear, since it would only report the switch you just set. An **attachment still counts as grounding** either way — attach a PDF with the database off and the reply is treated as grounded in it.
  - Nothing to do either way. If you left Speculation at its default (off), that behavior is now simply what Documents-on does, and nothing changes. If you used to switch it on, replies are unchanged but you will now see the marker where you previously saw nothing, and the AI is reminded to stick to what was retrieved. `raven-minichat`'s `!speculate` command is gone with it.

*Raven-server*

- the server log now records **every** request, at `--log-level DEBUG`, and records the *shape* of one rather than its content: counts, lengths, formats, durations, and the names of models, voices and filters. Running under `waitress`, a request that arrived and succeeded previously left no trace anywhere, so there was nothing to look at when a client seemed to be talking to nothing.
  - Your text, audio and images are not logged, at any log level. Two deliberate exceptions: `webfetch` records the URL it was asked to fetch, that being the request's identity and undiagnosable without it, and an error names the offending byte offset and codepoint rather than quoting the text around it. A `websearch` query is your question in your own words, so it is counted and not quoted.
  - The embeddings endpoint used to be the only one that said anything, printing a line and drawing a progress bar per request; speech-to-text drew one too. Both are gone — a progress bar is a display for one foreground job, and a server handling several callers at once has no foreground to draw in. Transcription instead logs how much audio it was given and how long it took.

*Raven-arxiv-download*

- a repeated identifier is dropped before any metadata is fetched, rather than being carried to the download step and skipped there. **The two arXiv tools now treat duplicates alike**: `raven-arxiv2bib` already discarded exact repeats as it collected its input, and `raven-arxiv-download` paid for one all the way to the download step. Two *versions* of one paper are not repeats in either tool, and still fetch as two — asking for v3 and v5 is asking for both.

*Raven-pdf2bib*

- extracts PDF text with the bundled `pypdf` instead of the external `pdftotext` (poppler-utils) binary — one fewer system dependency to install.

*Constellation-wide*

- Install (GPU): `torch` / `torchvision` / `torchaudio` are pinned as a matched set and installed as CUDA 12.8 (`+cu128`) wheels from a dedicated PyTorch index (`pytorch-cu128` in `pyproject.toml`), so a dependency re-lock can't silently swap in a mismatched-CUDA wheel that fails to load at import. These wheels run on both CUDA 12 and CUDA 13 driver stacks; macOS installs must remove that source first (see the README's CUDA section).

- **the file browser's Find field now searches the way the rest of Raven does.** Typing lowercase matches regardless of case, and typing any capital asks for an exact match — so `readme` finds `README.txt` while `README` finds only the shouted one. Several words are matched independently and in any order, anywhere in the name: `2026 report` finds `annual_report_2026.pdf`. This is the same search Raven-visualizer's search field and the graph viewer have always used.

- **the file browser's type filter can offer a named group of formats** instead of one entry per extension. Raven-librarian's attach dialog opens on *Documents and images*, which is every format it can actually take — 21 extensions, previously reachable only as *All files* with everything else mixed in — and offers *Documents* and *Images* for when you know which you want. Hovering the filter lists the extensions it covers.
  - The offered sets are asked for at startup rather than written down, so the picker cannot come to disagree with what Raven will accept.

**Fixed**:

*Raven-librarian*

- moving or renaming your documents folder no longer breaks the index. On the next startup scan, Raven compared each document against the *full path* it was indexed under, so a folder reached by a different route — renamed, moved, or through a symlink — looked like an entirely new collection: every file to be re-read, every indexed document to be dropped. It did not get that far either, aborting the scan with `'<file>' is not in the subpath of '<documents dir>'`. Documents are now matched by their path *relative* to the documents folder, which is what identifies them. Existing indexes are unaffected and need no rebuild.
- a document that is a symlink is now indexed instead of aborting the run. Opening a document store containing one failed outright with `'<target>' is not in the subpath of '<documents dir>'`, because each document's path was resolved through the link before being turned into an id relative to the documents directory — so a link pointing anywhere outside that directory took the whole store down with it, no document indexed. Symlinks now keep the path they were reached by, which is also the one the id is built from. This makes a document collection assemblable as a *view* over files that live elsewhere, without copying them.
- a malformed tool-call request from the language model no longer kills the whole reply with a `TypeError`. Raven builds a report for the AI in each of these cases — a request missing its type, naming no tool, or carrying unparseable arguments — and then crashed on the way to delivering it, so none of them ever reached the AI. The report now arrives as an errored tool result, which the AI can read and act on.
- the F8 hotkey (copy the chatlog to the clipboard) now works. It raised a `TypeError` and copied nothing; only the equivalent toolbar button worked, which is why the failure went unnoticed.
- a language-model backend error mid-reply (e.g. a broken model prompt template) no longer fails silently — the AI turn used to vanish with no reply and no on-screen indication. The failure now appears as an assistant message naming the backend error and its reason; reroll it to retry.
- a chat round whose AI turn finishes *instantly* — a backend error, most often — no longer renders the AI's message above your own message. The user turn and the AI turn ran as separate concurrent background tasks with no ordering guarantee between them; harmless while the AI took a second or more to produce its first output, but visibly out of order when it returned at once. The user turn now completes before the AI turn starts.
- models whose chat template requires all system messages to precede the first user turn (e.g. Qwen3.5) no longer fail every AI turn with a backend template error. The temporary context injects — current date and time, the focus-on-latest-input reminder, the answer-from-context-only reminder — sit after your latest message by design, so they now go out in the user role instead of as system messages.
- pressing Enter on an empty input in a chat you haven't written in yet no longer produces a reply in which the AI discusses its own instructions. Sending an empty message is still the way to let the AI take another turn — but that now requires something of yours to take it about; with nothing said yet, the only input the model had was Raven's internal per-turn notes, so it answered those.
- closing the window during startup is much more robust. A mid-boot close previously could hang the app or crash it (segfault), as deferred startup work and background tasks (the avatar renderer, the chat-view rebuild) run on background/callback threads and raced the GUI teardown. Shutdown now runs a deterministic two-phase sequence — cancel everything, drain it, then destroy the GUI context — and the racing operations bail once teardown has begun. (One rare crash path remains: a URL-heavy message still rendering in the vendored Markdown worker at the moment of close; tracked for a follow-up.)
- RAG: adding a new document to the database no longer logs a spurious `KeyError` traceback, and the indexing progress no longer counts a single new file as two changes. A new file's filesystem create+modify events could queue a *delete* for a document that was never indexed (more likely with larger files such as PDFs, which emit several write events while being copied in); the delete half of an update is now emitted only when the document actually exists.
- RAG: the documents directory is now auto-created at startup if missing. Previously, a fresh install or moved-aside docs dir crashed the app with `FileNotFoundError` from `inotify_add_watch` before the GUI came up.
- RAG: rapid sequences of file adds and updates no longer crash the indexing pipeline with `TypeError: string indices must be integers`. The pending-edits queue now stores delete entries in the same shape as add/update entries, so the dedup pass can run uniformly.
- the chat view no longer drags you back to the bottom while the AI is writing. Scrolling up to re-read something during generation now stays where you put it, across tool calls too — which on a thinking model is the difference between waiting and reading. The view follows new content only when you were already at the end when it arrived, and a reply finalizing or a tool result landing is treated the same way.
- the chat view now actually reaches the message you just sent. "Scroll to the end" read the panel's scroll maximum before the new message had been laid out, so it stopped where the *previous* message ended — on Send, the view typically stayed on the greeting.

*Raven-visualizer*

- with `raven-wos2bib`: BibTeX records with a brace in an unexpected field are no longer lost. The converter escaped some field values but not others, so a single `{` in, say, a DOI ended the field early and made the whole record unparseable — 8 records in 96296 on the Web of Science hydrogen corpus. Every field is now escaped. Existing `.bib` files converted with an earlier version keep the defect; reconvert them from the Web of Science source to recover the lost records.
- importer: a BibTeX record that fails to parse is now reported instead of silently missing from the dataset. The warning names the file, the record key, the line, and which field's braces look unbalanced, so the offending data can be found and fixed. Previously a record that lacked a title was reported but one that could not be parsed at all was not — the wrong way round, since the second is the case that leaves no other trace.
- importer: and where the culprit is a stray brace in the text — mathematics that reached the `.bib` through a PDF extractor, most often — the record is now **recovered** rather than only reported, and joins the dataset as though it had parsed. Only records that already failed are touched, so nothing that reads correctly today can be affected. Your `.bib` file is not modified: this repairs Raven's reading of it, and `raven-fixbib` is the tool that writes a repair back.
- importer: BibTeX case-preservation grouping braces (`{Word}`, `{ACRONYM}`, `{{nested}}`) are now stripped from titles and abstracts, and common LaTeX diacritics (`\"o` → ö, `\'e` → é, `\c{c}` → ç, `\ae`, `\o`, …) are rendered as Unicode. Escaped literal braces (`\{`, `\}`) are preserved.
- importer: **author names** now get that same treatment, which they were missing — a citation that read `H{"a}m{"a}l{"a}inen and Erkkil{"a}` now reads `Hämäläinen and Erkkilä`. It affected only names carrying diacritics, so it hit Nordic, German, Polish and Central European authors and left English ones alone (4% of names in a mixed-language mechanics bibliography). The raw BibTeX field is stored alongside, so export is unchanged. With *Raven-librarian*, whose attribution line for pasted BibTeX uses the same formatter.
  - Fixed in the same pass: the space-separated spelling of the letter-named accents (`{\c e}`, `{\k a}`, `{\v s}`) was left as literal text. Only the braced form (`\c{e}`) was recognized, though a LaTeX control word ends at the first non-letter and both are equally valid — and `.bib` files favour the space form, since the idiom wraps the whole accent in a case-protecting group.
- opening a dataset while the mouse is over the semantic map no longer crashes the tooltip background task with `AttributeError: name 'kdtree' is not defined`. The new dataset was published to the app one field at a time, so a concurrent tooltip render could read it half-built; it is now assembled fully before becoming visible.
- opening a dataset while the tooltip, info panel, or word cloud is rebuilding can no longer raise an `IndexError` from indices computed against the old dataset landing on the new one. Each background render now pins one dataset snapshot for the whole build instead of re-reading the shared reference as it can change mid-build.
- running without *Raven-server* (which the Visualizer treats as optional, loading models locally) no longer dumps a multi-frame connection-refused stack trace to the log for every model load. The low-level reachability probe now logs a single concise line at debug level; callers that require the server still report a clear error.

*Raven-server*

- the console no longer prints the text sent for sentiment classification, nor the classification of it. That endpoint is fed whatever is being said in the chat — so on a session with the AI avatar running, the conversation was being written to the server's console and to any log capturing it, at every log level and with nothing to switch it off. Only the length is recorded now, at debug level.

*Raven-avatar*

- an AI reply ending on a stray Markdown bullet (`...to the naked eye:` followed by a lone `*`) no longer aborts the whole spoken utterance. Such a fragment reached the speech synthesizer as a sentence of its own, with nothing pronounceable in it, and the resulting zero-length audio crashed the encoder; fragments with no speakable content are now dropped before synthesis, and empty audio encodes cleanly rather than failing.
- the log no longer fills with the avatar announcing, every three seconds for as long as the session is idle, that it is returning to its neutral expression. It says so now only when it actually returns from one — so on a quiet session the line is silent, and when it appears it means something. *Raven-server*'s console had the matching half of this, dutifully reporting each of those re-assertions as it applied them; it too now speaks only on a change, and names what the emotion changed from.
- releasing an avatar instance that the server does not have no longer fails. Leaving a client running across a *Raven-server* restart meant the client still held an instance ID from the previous server process, so closing it raised a 500 and a traceback on the server console on the way out. Unloading now succeeds whether or not the instance is there — the point of the call is that the instance is gone afterwards, and one that was never there satisfies that already.

*Raven-cherrypick*

- triaging an image (cherry / lemon / winner) while its mips were still loading no longer leaves it stuck at a reduced resolution or failing to appear. The triage move relocates the file out from under the in-flight background decode, which then failed with `FileNotFoundError`; the load now restarts from the file's new location, whether it was filling in the full-res level of a preloaded image or doing the initial decode of a cache-miss one.
- triaging then immediately navigating (e.g. `C` then `Right` within one ~16 ms frame) no longer tags the wrong image. DearPyGui dispatches same-frame key presses by keycode, not by press order, so navigation (lower keycode) moved the current image before the triage key read it; keyboard navigation is now deferred by one frame so a same-frame triage key acts on the intended image.
- in a filtered view (e.g. neutral-only), tagging the current image then pressing a navigation key no longer skips the image that took its place. The just-tagged image leaves the filtered set, and navigation was resolving from the nearest surviving tile and then stepping again, overshooting by one.
- stepping off a grid row edge (`Right` from the last tile of a row, `Left` from the first) no longer reloads the next image's mips from scratch. The neighbor preload cache clipped its horizontal reach to the current row, so the row-wrap target — the actual `Left`/`Right` destination at an edge — was never prefetched; the horizontal reach now follows the linear navigation order across row boundaries.
- navigating to a preloaded neighbor no longer flashes it at low resolution before sharpening. Speculative preloads were capped at a fixed quarter resolution, so even on a cache hit the larger mip levels were regenerated on arrival — a visible re-sharpen on every step. The cap is now adaptive to the current zoom: a neighbor is prefetched at exactly the resolution the pane shows it at, so small images at fit-zoom arrive crisp while multi-MP photos still avoid the slow full-res GPU→host readback they don't need.
- a thin colored stripe no longer appears along the image edge when zoomed in past 1:1. Under magnification the GPU's bilinear sampler read just past the texture boundary and wrapped to the opposite edge (a bright bottom row bleeding into the top, etc.); the sampled region is now inset by half a texel when magnifying. Unchanged at 1:1, which samples exactly on the texel grid.

*Raven-arxiv-download*

- downloaded-filename titles no longer read as run-on sentences. Clause boundaries that the filename sanitizer used to drop (`:` `?` `!` `;` followed by a space) now become ` - `, em/en dashes become a plain `-` instead of collapsing to a double space, and a compound-joining `/` becomes `-` instead of mashing the two sides together (`Twitter/X` → `Twitter-X`, not `TwitterX`). Example: `…Own Exploration? Gradient-Guided…` → `…Own Exploration - Gradient-Guided…`.
- a nonexistent or malformed arXiv ID (e.g. a typoed month, `2614.19062`) now fails with a readable one-line "no arXiv entry for ID …" message — no traceback, since it's an expected user error — instead of an opaque `AttributeError`. The run continues to the remaining IDs; genuinely unexpected errors (network, parse bugs) still print a traceback for debugging.
- with *raven-arxiv-search*: HTTP 429 responses from the arXiv API no longer abort the run. Both tools now retry up to three attempts with backoff (honoring `Retry-After` when set, else exponential 3/6 s) and send an identifying `User-Agent` per arXiv's API TOU. Triggered occasionally on cache-miss bursts even when the caller is within the published 3 s rate limit; `raven-arxiv-download` also now goes straight to HTTPS instead of getting redirected from HTTP.

*Constellation-wide*

- All client HTTP calls to *Raven-server* and to the LLM backend now use connect/read timeouts, so a server or backend that becomes unreachable mid-connection fails fast instead of hanging indefinitely. Matters most when either is configured to run on another machine that is down. Timeouts are configurable in `raven.client.config` (`network_timeout`) and `raven.librarian.config` (`llm_network_timeout`); streaming endpoints bound only the connect, leaving long-lived streams unbounded.
- `dpg_markdown` bullet lists and blockquotes now render correctly inside tooltips (and any other initially-hidden container). Previously every bullet glyph in a tooltip stacked at the top-left, because DPG reports `get_item_pos() == (0, 0)` for children of a hidden container; the bullet drawlists are now deferred until their row has been laid out.
- `deviceinfo.validate`: `device_name` label now reflects the actual running backend. Previously a working MPS / XPU / Vulkan setup was logged as `'CPU'` in the startup "Compute device for ..." line because the labeling block was tied to a CUDA-prefix check it shouldn't have been. Cosmetic — the actual compute device was always correct.
- every GUI app used to open with four `DeprecationWarning`s about `add_font_range`, a DearPyGui call that stopped doing anything in DPG 2.3, which builds font atlas character ranges by itself. Raven no longer declares ranges, and now requires `dearpygui>=2.3` (it was `>=2.0.0`) so that it doesn't have to. Text rendering is unchanged — a character that came out as a box before still does, and still means the font lacks that glyph.
- the file browser's type filter now matches extensions regardless of case, so a photo named `SCAN.JPG` appears under a `.jpg` filter instead of vanishing from it.
- on Linux and macOS, the file browser's shortcuts panel no longer lists raw block devices alongside the real mount points. It scanned `/dev` for anything named `sd…` or `nvme…` and offered each as a destination — on a plain single-disk machine, four extra entries naming the same disk in four ways, none of them a directory, so clicking one could only produce "the selected item is not a directory". Windows was never affected.

---

**0.2.7** (22 April 2026):

**Added**:

- New submodule: `raven.papers` — consolidates all paper and bibliography tools.
  - New tool: `raven-arxiv-search`.
    - Usage: `raven-arxiv-search query.txt -o sometopic.bib`
    - Search arXiv with boolean expressions (AND/OR/ANDNOT, quoted phrases, parenthesized grouping), export results as BibTeX.
    - This was originally a standalone tool, [`arxiv-api-search`](https://github.com/Technologicat/arxiv-api-search).
  - New CLI option: `raven-arxiv-download --from-bib sometopic.bib -o papers/`.
    - Does as it says on the tin.
    - Takes the arXiv ID for each BibTeX entry from the `eprint` field.
      - Metadata records returned by arXiv's API (e.g. via `raven-arxiv-search`) include that field.
      - Additionally, if the `archiveprefix` field is present, it is checked that its value is `arxiv` before attempting to download the paper.
  - Internal changes:
    - Relocated from `raven.tools`: `raven-arxiv2id`, `raven-arxiv-download`, `raven-burstbib`, `raven-csv2bib`, `raven-pdf2bib`, `raven-wos2bib`. CLI command names unchanged.
    - Shared `RateLimiter` (thread-safe, tqdm progress bar) — extracted from the arXiv downloader, now also used by the search tool.
    - Shared `bibtex_escape`/`bibtex_unescape` — single source of truth, replacing duplicate definitions in `csv2bib` and `wos2bib`.
    - Consolidated arXiv ID handling: `identifiers.strip_version()` replaces three separate implementations.
    - New dependency: `feedparser>=6.0`.
- *Raven-cherrypick*: new "mark winner" action (Ctrl+Shift+C, or Ctrl+Shift+click cherry button). Marks the current image as cherry and all other selected images as lemon — one keystroke to commit a compare-mode choice.
- HTTP API: new `/api/stt/info` and `/api/tts/info` endpoints return the currently loaded model name and the model's native sample rate. Lets clients avoid hardcoding values that drift when the server config changes.
- *Raven-avatar*: client-side crop.
  - New crop panel in the settings editor — drag a rectangle on a viewport overlay, debounced push to the server, live preview. Works on the rendered avatar, independent of upscaler and postprocessor.
  - Server-side: crop now happens *before* upscale in the pipeline (previously after), so the upscaler only processes the cropped region.
  - Avatar renderer now sizes its texture reactively from decoded frame dimensions (previously fixed-size, mismatched when the server changed resolution mid-session).
  - New server-side telemetry: `X-Server-Stats` response header with per-request server time, so the client can display an end-to-end latency breakdown.
- *Raven-avatar*: avatar apps (settings editor, pose editor) now show the Raven version in the viewport title.
- *Raven-avatar*: settings editor now has per-parameter help as Markdown tooltips, sourced from each filter's docstring. An info button next to every postprocessor parameter shows that parameter's description, rendered via `dpg_markdown`. Filter-level info buttons show each filter's preamble. New helper module `raven.common.docstring_utils` parses Raven-style docstrings (`` `name`: description``) into summary + per-parameter sections.
- *Raven-avatar*: pose editor F1 help card with a prose section explaining the posing workflow. Hotkey table + two-column layout matching the settings editor / xdot viewer style.

**Changed**:

- *Raven-avatar* performance improvements:
  - ~4–5% faster avatar rendering via `torch.inference_mode`, cached `affine_grid` base grids in the THA3 engine, and zero-copy pose tensor expansion. Pure inference paths across the render pipeline (avatar, postprocessor, upscaler, pose editor) now use `inference_mode` instead of `no_grad`.
  - Chromatic aberration filter optimized, now 2.2× faster (batched grid_sample and GaussianBlur calls), cached grids, in-place alpha averaging.
    - Default postprocessor chain now 38–67% faster depending on resolution.
  - Auto-sized GaussianBlur kernels based on sigma (previously hardcoded at maximum). Saves blur cost at typical settings (e.g. CA at sigma=1.0: kernel 5 instead of 13).
  - Anime4K upscaler — eliminated unnecessary tensor clone, in-place output clamp.
  - New upscaler quality options: `bilinear` and `bicubic` bypass Anime4K entirely for compute-constrained GPUs.
    - This trades off image quality for ~18× faster upscaling - which may upgrade an avatar from 20 FPS to 25 FPS.
    - Quality difference is unnoticeable with the postprocessor enabled (with the default chain).
    - Main difference between Anime4K and `bicubic` is in details with thin lines, such as the rims of a character's glasses.
  - `bicubic` is now the default upscaler. The quality is good enough with the default smoke and mirrors enabled (and can now get 25 FPS at 1024x1024 on a laptop RTX 3070 Ti).

- *Video processing* (`raven.common.video`):
  - `chroma_subsample` filter to simulate a lo-fi video look.
    - Reduces chrominance (color) resolution while keeping luminance (brightness) at full resolution. Real video systems use this to improve compression, because human vision isn't as sensitive to color as it is to brightness.

- *Speech TTS wire format* default: MP3 → FLAC. FLAC is lossless, so the remote path now produces bit-identical audio to the local path; MP3's historical advantage (smaller files over the wire) doesn't matter on the trusted local network Raven targets. MP3 was originally chosen because Kokoro-FastAPI couldn't produce FLAC reliably; Raven no longer routes through Kokoro-FastAPI. The OpenAI-compatible `/v1/audio/speech` endpoint stays on MP3 (SillyTavern-facing; OpenAI spec defaults to MP3). Callers can still request any PyAV-supported format explicitly via `format=`.

- *Speech STT wire format* (client → server): MP3 → FLAC, for symmetry with the TTS direction and for the same reason — lossless on a trusted LAN beats lossy. `raven.client.api.stt_transcribe_array` now encodes the audio as FLAC before upload; the server continues to auto-detect the container format via PyAV, so no server-side change was needed.

- New module `raven.common.audio.resample` — device-agnostic sample-rate conversion (torchaudio-backed). Works on numpy arrays and torch tensors; three quality presets (`"default"`, `"kaiser_fast"`, `"kaiser_best"`) matching librosa's naming.
  - New dependency: `torchaudio>=2.4.0`.

- *TTS/STT* plumbing improvements:
  - New module `raven.common.audio.speech.stt` — Whisper wrapper callable in-process (no Flask).
    - `raven.server.modules.stt` is now a thin wrapper that decodes the audio container and forwards to the common layer.
  - New module `raven.common.audio.speech.tts` — Kokoro wrapper callable in-process (no Flask).
    - Two-layer API: `synthesize_iter` yields per-segment `TTSSegment` with already-absolute word timestamps, `synthesize` is the concatenating wrapper returning a single `TTSResult`.
    - `raven.server.modules.tts` is now a thin wrapper that casts float→s16 at the transport boundary, URL-encodes Unicode phonemes for HTTP headers, and handles Flask response construction.
  - New module `raven.common.audio.speech.lipsync` — engine-agnostic lipsync and subtitle driver.
    - Pure time-slicing for phoneme and word tracks, plus a callback-driven tick loop (`drive(on_tick, clock, tick_seconds)`).
    - Consumers compose tracks inside their own `on_tick` closure, calling `phoneme_at(stream, t)` / `word_at(timings, t)` as needed — lets the same loop drive avatar morphs, per-phoneme subtitles, word-level captions, or any combination.
    - No dependency on Kokoro or any other TTS engine.
  - New module `raven.common.audio.speech.playback` — synchronous audio playback + optional lipsync drive.
    - `play_encoded` and `play_encoded_with_lipsync` factored out of `raven.client.tts`; callers wrap in their own task manager for fire-and-forget.
    - Pure: takes the `player` as an explicit argument; the avatar-driving closure is caller-supplied.
  - `raven.client.tts` gains `play_encoded_with_avatar_lipsync` — the avatar-specific Raven wrapper that builds the mouth-morph closure and handles the server-side `avatar_modify_overrides` cleanup. Used by both `api.tts_speak_lipsynced` and `MaybeRemote.TTS.speak_lipsynced` local mode.

- `raven.client.mayberemote` new services, mirroring the existing `Dehyphenator` / `Embedder` / `NLP` pattern.
  - `TTS` and `STT`. Apps can now use speech locally when the server is down (or skip the round-trip entirely for latency), with a uniform API across modes.
    - `STT.transcribe` auto-resamples mismatched input.
    - `TTS.synthesize(format=...)` is shape-agnostic: no argument returns float32 `TTSResult` in both modes; `format="flac"`/`"mp3"`/… returns `EncodedTTSResult` ready for playback or storage.
      - Caching lives in the bottom layers (one source of truth per (location, shape)), so the mayberemote dispatcher has no cache state of its own.
    - `TTS.speak` / `TTS.speak_lipsynced`, mirroring `raven.client.api.tts_speak*`. Local-mode TTS; and local playback + remote avatar (for lipsynced).
      - `prep` accepts either `TTSResult` or `EncodedTTSResult` — encoded to FLAC internally as needed.
    - Stop / query playback via the player, not TTS: `raven.common.audio.player.instance.stop()` / `.is_playing()`. One call surface works for all three API paths (explicit-local, explicit-remote, maybe-remote), since the audio hardware is always client-local regardless of where synthesis happens.
    - `DPGAvatarController` now routes synthesis + playback through `MaybeRemote.TTS` (instead of the explicit-remote `api.tts_prepare_cached` / `api.tts_speak_lipsynced`). Stop goes through the player as above. Reads `tts_allow_local` / `tts_model_name` / `tts_lang_code` from `raven.client.config` and the device from `client_config.devices["tts"]`; flipping `tts_allow_local = True` gives the app standalone TTS capability (Kokoro loaded in-process when the server is unreachable).
  - New `raven.client.config.devices` — same shape and convention as `devices` in `raven.{librarian,visualizer}.config`. Validated by `raven.common.deviceinfo.validate` during `api.initialize` (CUDA → CPU fallback, `device_name` injection). Currently holds the `tts` record; more services join as their `<svc>_allow_local` paths gain real use.
  - `Classifier` (text sentiment), `Translator` (machine translation), `Postprocessor` and `Upscaler` (imagefx).
    - Each dispatches to the corresponding Raven-server module in remote mode and to a local in-process instance in local mode, with identical call surfaces.
    - `Translator` takes a `spacy_model_name` for local-mode sentence chunking.
    - `Upscaler` caches local `_LocalUpscaler` instances per `(width, height, preset, quality)` config, since Anime4K model choice depends on preset/quality and the constructor loads real weights.
  - With these, now every server module that isn't license-constrained (avatar, websearch) is reachable via `MaybeRemote` - and the same functionality is transparently available in-process in local mode.

- *`raven.client.api` init guard* collapses from a four-line `if not util.api_initialized: raise RuntimeError(...)` block at every function head to a one-line `util.require()`. New `raven.client.util.require()` (re-exported as `raven.client.api.require`) mirrors `raven.common.audio.{player,recorder}.require()`. 40 call sites rewritten; message now routes through the stack trace rather than prefixing each error with the function name.

- *TTS warmup* gains a common-layer implementation and routes through `MaybeRemote.TTS.warmup(voice)`, matching the three-layer shape of `synthesize` / `speak`. New `raven.common.audio.speech.tts.warmup(pipeline, voice)` runs the throwaway synthesis in-process; `raven.client.tts.tts_warmup(voice)` stays as the explicit-remote path; `MaybeRemote.TTS.warmup` dispatches. Raven-librarian now warms up via its avatar controller's TTS dispatcher, so standalone runs (`tts_allow_local=True`) warm the local pipeline instead of hitting the server.

- *Audio player / recorder singletons* lifted out of `raven.client` into `raven.common.audio`. `Player` / `Recorder` live next to their own classes now, not inside the remote-API config namespace. Apps that need audio call `raven.common.audio.initialize(player=..., recorder=...)` — each side accepts `True` (defaults), a kwargs dict, or `False` to skip. `raven.client.api.initialize` no longer touches audio; apps that don't use audio (e.g. `raven-importer`, `raven-dehyphenate`) therefore skip the pygame/pvrecorder init entirely. Downstream consumers read `raven.common.audio.player.instance` / `recorder.instance` (or `.require()` for fail-fast when uninitialized).
  - `raven.client.api.tts_stop` / `tts_speaking` and `MaybeRemote.TTS.stop` / `.is_speaking` removed; use `raven.common.audio.player.instance.stop()` / `.is_playing()` directly. Rationale: the three API paths (explicit-local / explicit-remote / maybe-remote) all share the same local audio hardware, so one call surface works for all of them.
  - `api.initialize` signature lost its `tts_playback_audio_device` / `stt_capture_audio_device` arguments — those belong to `raven.common.audio.initialize` now.

- *Image codec*: new module `raven.common.image.codec` with `encode` / `decode` — the unified image I/O layer. Parallel to `raven.common.audio.codec`.
  - Lifts the previously-duplicated decode logic out of `raven.server.modules.imagefx` (AGPL, this module 100% by @Technologicat) and `raven.common.image.utils` (BSD) into a single BSD-licensed home.
  - `decode` accepts bytes, binary streams, or filesystem paths interchangeably, and returns natural channel count (no forced RGBA).
  - Callers that need a guaranteed 4-channel output use the new `raven.common.image.utils.ensure_rgba` helper.
  - `IMAGE_EXTENSIONS` moved here from `image.utils`.

- *XDot viewer*: dense graphs no longer burn CPU while the cursor is outside the widget. The hover-refresh path was unconditionally marking the frame dirty every tick, which defeated the idle throttle; now the flag is only raised when hover state actually changes. Visible on graphs with many edges — idle FPS drops back to the background rate instead of pegging at the redraw rate.

- *Raven-avatar*: the "data eyes" effect fadeout moved from the client to the server. The client now sends one `start_data_eyes` / `stop_data_eyes` command; the server's animator cycles the cels and drives the fadeout like the other animation drivers. New HTTP endpoints `/api/avatar/start_data_eyes` and `/api/avatar/stop_data_eyes`, and a new animator setting `data_eyes_fadeout_duration` (default 0.75 s) alongside the existing `data_eyes_fps`. The former `avatar_modify_overrides({"data1": ...})` fade stream (~45 HTTP calls per fadeout at 0.75 s) is gone.

- *Raven-server* / *natlang*: `/api/natlang/analyze` now returns language-neutral JSON instead of a spaCy `DocBin` binary blob. Each response item is `{"lang": ..., "doc": <spaCy Doc.to_json()>}`, with optional `"vectors"` when the new `with_vectors` request flag is set. Per-item `lang` makes the wire format naturally multilingual-ready (for future server configurations loading multiple pipelines — e.g. English plus Finnish). Python clients remain unaffected at the API surface — `raven.client.api.natlang_analyze` continues to return `list[Doc]` — but non-Python clients (a future JS avatar frontend) can now consume the endpoint directly. Trade-off: the DocBin vocab-sharing optimization is gone, so repeated categorical strings (POS tags, dep labels, lemmas) appear once per token rather than once per batch; invisible in practice given Raven's LAN-only deployment (KB-range payloads on localhost or trusted LAN). `with_vectors=True` round-trips `doc.tensor` as base64 float32, giving `MaybeRemote.NLP` callers identical feature parity across local and remote modes.

- *Common utilities*: minimum `unpythonic` dependency bumped to 2.1.0. `environ_override`, `maybe_open`, `UnionFilter`, and `si_prefix` graduated to `unpythonic` in that release — Raven's local copies have been removed; the names now come from `unpythonic`.
  - Visible side effect: SI-prefixed numbers in log messages (bitrate, byte-rate, pixel-rate strings in the avatar renderer and audio codec) now use correct SI casing — lowercase `k` for kilo (previously uppercase `K`, which is the symbol for kelvin). `si_prefix` also gained binary (base-1024) mode, sub-unity prefixes (`m`, `µ`, ...), and correct handling of negative and zero values.

**Fixed**:

- *Raven-minichat*:
  - `raven-minichat` no longer crashes on MS Windows. Previously, the command would fail at startup with `ImportError: No module named 'readline'` because Python's stdlib `readline` module is POSIX-only. The fix is a three-tier hybrid load: try stdlib `readline` first (Linux/macOS), fall back to `pyreadline3` (a drop-in Windows replacement; `pip install pyreadline3` to get the full experience), and finally degrade gracefully to plain `input()` if neither is available — the chat loop still works, you just lose command history, tab completion, and persistent cross-session history. When running in the degraded mode, a startup notice explains what's missing and how to restore it.

- *Numerical utilities* (`raven.common.numutils`):
  - `psi()` (mollifier helper, also used via `nonanalytic_smooth_transition()`) no longer emits a stray `RuntimeWarning: divide by zero encountered in divide` when evaluated at `x = 0`. The function is correct — it uses the standard "compute-then-mask" idiom `np.exp(-1.0 / x**m) * (x > 0.0)` where `-1/0 = -inf`, `exp(-inf) = 0`, and the mask zeros the result — but numpy was still emitting the warning from the division step. A previous suppression attempt used `warnings.filterwarnings(..., module="__main__")`, which silently failed in practice (numpy emits the warning from its own internal module, not `__main__`). Replaced with `np.errstate(divide='ignore', invalid='ignore')`, numpy's own mechanism for suppressing float-error warnings within a dynamic extent.

- *NLP tools* (`raven.common.nlptools`):
  - `count_frequencies(..., lemmatize=False)` no longer crashes with `TypeError: 'int' object is not callable`. Latent bug: the non-lemmatize branch called `.lower()` on a spaCy `Token`, whose `.lower` attribute is the orth hash (an integer), not a method. The default `lemmatize=True` path rebinds the loop variable to `token.lemma_` (a `str`) before calling `.lower()`, so the default masked the bug. Every caller threading `lemmatize=False` through crashed.

- *CSV parsing* (`raven.common.readcsv`):
  - `parse_csv` with autodetected headers no longer silently drops the first data row. Latent bug: after sniffing the header via `next(reader)`, the code rebuilt a fresh `csv.reader` at the current (post-header) file position and then advanced another row via `reader.__next__()`, so the main parse loop effectively started at row 3. A `header + N`-row file returned `N - 1` rows; a `header + 1`-row file silently returned `[]`.

- *Audio codec* (`raven.common.audio.codec`):
  - `decode` no longer crashes on FLAC (and any other container that reports `duration=None` via pyav). A log-info line divided `None / av.time_base` on first frame, raising `TypeError`. Affected any caller decoding FLAC from a `BytesIO`.

- *arXiv tools* (`raven.papers`):
  - `raven-arxiv2id` (and other tools using arXiv ID extraction from filenames): fix detection of IDs embedded between underscores, letters, or hyphens in filenames. Previously, filenames like `Smith_2301.12345_notes.pdf` silently failed to match. Also adds support for 4-digit new-style IDs (2007–2014 era, e.g. `0704.0001`) and old-style IDs with subject class prefix (pre-2007, e.g. `hep-th/0601001`).

- *BibTeX tools* (`raven.papers`):
  - Fix `bibtex_escape`: unmatched `{` in source text (e.g. WoS abstracts) produced unbalanced braces that broke bibtexparser parsing. The old approach doubled braces (`{` → `{{`); now uses proper LaTeX escapes (`\{`, `\}`).
  - Add missing `#` and `$` escaping — both are BibTeX/LaTeX specials that could cause parse or render errors in downstream tools.
  - `pdf2bib` now applies `bibtex_escape` to all field values (literal, LLM-extracted, and function-generated). Previously, LLM output was written unescaped.
  - `requests` and `tqdm` added as explicit dependencies (were used directly but only present as transitive deps).
  - `raven-csv2bib` now converts **all** input files when given more than one. Previously, entries from all but the last file on the command line were silently dropped — the aggregation loop collected rows into an accumulator that a later loop never read, so only the last file's entries made it into the output.

- *Video processing* (`raven.common.video`):
  - Fix filter cache invalidation on resolution change. Filters using texture caches now check their own tensor dimensions instead of relying on the video frame dimensions, preventing stale data when the image resolution changes mid-session.

- *Raven-avatar*:
  - Settings editor: avatar panel now resizes with the window (especially noticeable when going fullscreen). Previously the rightmost column expanded uselessly; now the postprocessor column stays at its default width and extra space goes to the avatar panel.
  - Fix settings editor crash when loading filters with `!ignore` parameters (e.g. anything with a `name`). The canonize and generate paths now skip these, matching the GUI build path.
  - Pose editor: FPS counter no longer shows near-zero on the first frame (warmup fix), and an idle throttle brings CPU use down when the editor is not being interacted with. The window now also fits on 1080p displays without overflow.

- *Text file I/O* (Windows correctness): every text-mode `open()` in Raven now specifies `encoding="utf-8"` explicitly. On Linux/macOS the system default is UTF-8, so this was latent; on MS Windows Python's default is the ANSI code page (cp1252 in Western locales), silently corrupting non-ASCII content — emotion names with Unicode symbols, API keys with high-bit bytes, BibTeX entries with accented author names, audio timing reports with paths containing ä/ö. Affected `raven.common.readcsv`, the avatar pose / settings editors, `raven.server` (API key file, animator settings, emotion defaults), and `raven.papers.pdf2bib`. Also future-proofs for Python 3.15, which will start warning on a missing `encoding=` (PEP 597).


---

**0.2.6** (9 April 2026):

**Added**:

- New GUI app: *Raven-cherrypick*.
  - An image triage tool for quickly sorting a folder of images into cherries (keepers), lemons (rejects), and neutral.
  - Start with `raven-cherrypick some/path/to/images/`. If no path given, defaults to CWD.
  - GPU-accelerated Lanczos scaling with mipmapped progressive loading and preload cache for instant image switching.
  - No on-disk thumbnail cache, no metadata files. Image state is encoded by directory path (`base/cherries`, `base/lemons`).
  - Easy two-hand operation: arrows navigate; X=lemon, C=cherry, V=clear mark. Ctrl+click in grid view for multi-select.
  - Filter view: show only cherries, lemons or neutral, or show all (G / Shift+G to cycle).
  - Jump to next cherry/lemon/neutral with B/N/M.
  - Zoom/pan preserved when switching between images with the same dimensions — useful for comparing variations of the same shot.
  - Compare mode: select 2–9 images and press Enter to cycle through them automatically. Adjustable speed, pause/resume, zoom while cycling. Press a digit key to pick a winner and exit.
  - Status bar: current position, image dimensions and approximate aspect ratio, selection count.
  - F11 fullscreen mode and F1 help card available.

- New CLI tool: *Raven-conference-timer*.
  - A large-font countdown timer for conference presentations.
  - Start with `raven-conference-timer 15:00` (or bare minutes: `raven-conference-timer 15`).
  - Auto-sizes the window to fit the countdown text. `--size N` sets font size in pixels (default 500).
  - Color changes at configurable thresholds: white → yellow → red → pulsating glow when expired.
    - `--yellow` and `--red` set the thresholds (default 5:00 and 2:00).
  - Hotkeys: Space to pause/resume, F11 for fullscreen, F1 for help card, Esc to exit.

- *Raven-xdot-viewer*:
  - GUI: combobox to choose which GraphViz layout engine to use, re-rendering the current graph with the chosen engine.
  - Tooltip support for node annotations (from GraphViz `tooltip` attribute; e.g. Pyan3 2.4.0+ generates these).
  - Dashed and dotted edge rendering.
  - Error dialog for failed graph loads.
  - Idle framerate throttle (reduced GPU usage when not animating).

- *Raven-avatar*:
  - F1 help card for the avatar settings editor.

- *Video processing* (`raven.common.video`):
  - New filter: VHS head switching noise (horizontal distortion bands at frame bottom). The most iconic VHS artifact.
  - New noise mode: VHS, with PAL and NTSC modes. NTSC comes with 4:2:0 chroma subsampling for a more authentic analog look.
  - Bloom filter: added `sigma` parameter for controlling glow width. Recommended values: 7.0 for dreamy early 2000s anime glow, 1.6 for modern tighter glow.

- *Common libraries*:
  - New module: `raven.common.image` (image utilities and GPU-accelerated loader pipeline).
  - Extracted `SmoothValue`/`SmoothInt` into `raven.common.smoothvalue` (shared across xdot viewer, cherrypick, and future apps).
  - `PyTurboJPEG` dependency added for fast JPEG decoding. Requires the `turbojpeg` system-level library (on Debian-based Linux: `sudo apt install libturbojpeg`).

**Changed**:

- *Video processing* (`raven.common.video`):
  - There are now two noise stages: `noise` (sensor/film grain, early in the chain) and `analog_vhs_noise` (VHS tape noise, later). This better models the physical signal path.
    - If you have custom chains that use `noise`, check whether you need both stages.
  - Split `desaturate` into `desaturate` (retouching stage) and `monochrome_display` (display output stage), allowing separate control over the artistic and output desaturation.
  - Renamed the `translucency` filter to `translucent_display` for consistency with the new `monochrome_display`, and moved it late in the chain because it models a scifi display device.
    - If you have custom avatar postprocessor chains (in `raven.server.config` or custom JSON presets), rename the filter in your chain. The bundled presets have been updated.

- *Dependencies*:
  - Bump `mcpyrate` to 4.0.0.
  - Bump `unpythonic` to 2.0.0.
  - Widen Python support to `<3.15`.
    - But narrow `requires-python` to `<3.13` for `kokoro`/`misaki` compatibility.

**Fixed**:

- Compatibility: detect "Item not found" across different Python/DPG versions, needed in GUI code.

- *Raven-xdot-viewer*:
  - Fix dark-mode text contrast on colored node fills (text now adapts based on perceived luminance).
  - Fix graph area too small on 1080p displays.
  - Fix `--size` flag for fonts smaller than the default 120px.


---

**0.2.5** (3 March 2026):

**Added**:

- New GUI app: *Raven-xdot-viewer*.
  - This is a utility app for viewing GraphViz graphs (`.dot`, `.gv`, `.xdot`).
  - Start with `raven-xdot-viewer`.
  - In a future version, this technology will be deployed in Librarian for navigating the nonlinear chat history.

- Tools:
  - New command-line tool: *Raven-csv2bib*.
    - This converts comma-separated values (`.csv`) to BibTeX.
    - The first row of the `.csv` file must consist of column headers. Fields with these names will be populated in the BibTeX output.
      - For use with *Raven-visualizer*, the fields *Author*, *Year*, *Title* are required, and the field *Abstract* is optional.
        - If your dataset has no meaningful text descriptions beyond an item title, you can omit the whole *Abstract* column.
        - But if you have text descriptions, including them should improve the accuracy of the semantic map, by making it easier for Raven to detect which items are semantically similar.
      - Arbitrary other fields can be included and will be transcribed into the output BibTeX.
    - Author names use BibTeX format.
      - If an item has multiple authors, separate them with the lowercase literal word "and".
      - Each author name can have up to four parts (first, von, jr., last).
      - Each author name must be in one of three formats:
        - First von Last ("First Last" if no "von" part)
        - von Last, First ("Last, First" if no "von" part)
        - von Last, Jr., First
      - For more details, see: https://www.bibtex.com/f/author-field/

**Changed**:

- Bump minimum **Python** version to **3.11**.
  - **Upgrading to Raven 0.2.5 requires a fresh reinstall**.
    - To do this, delete the `.venv` hidden subdirectory inside your top-level Raven directory, then `pdm install`.
    - It should still remember if you had CUDA enabled, and if so, automatically install the NVIDIA packages.
  - Raven requires [`av`](http://pyav.org/docs/stable/) for its audio handling.
    - Particularly, the audio encoder for transporting TTS audio over the network from *Raven-server* to *Raven-librarian* needs `av`.
    - Recent versions of `av` may require installing some upgrades.
      - FFMPEG 7 is now required. Older versions (4, 5, 6) are no longer supported.
        - To see which version you have, `ffmpeg -version` (note only one dash).
        - For Ubuntu 22.04 based systems (e.g. Linux Mint 21.3), there is a PPA; for how to add it, see e.g. [here](https://blog.programster.org/install-ffmpeg-7-on-ubuntu-22).
      - To be able to build `av`:
        - Beside `ffmpeg` itself, you will need the various related `lib*-dev` packages from the same repository. You can use the `synaptic` GUI to locate them easily (filter the view by *Origin*).
        - From the distro's default repositories, you'll need `clang`.
        - You'll also need `cython`. For this, `pip install cython` (in the environment where you're running `pdm`) should work.
  - The Python upgrade was needed to support the [Hindsight](https://hindsight.vectorize.io/) AI agent memory system that will be later used by *Raven-librarian*.

---

**0.2.4** (16 December 2025):

**Added**:

- Tools:
  - New command-line tool: *Raven-arxiv-download*.
    - This takes arXiv paper IDs from the command line (e.g. 2511.22570, 2411.17075v5, cond-mat/0207270, math/0501001v2), and downloads the corresponding PDFs.
      - Main use case is: people have recommended a bunch of interesting arXiv papers, hence you have dozens of URLs or arXiv IDs on your phone, and you'd like to download the fulltexts on your PC.
      - The files are automatically named based on the metadata record queried from the arXiv API.
      - You can download either the latest version of each paper (default) or a specific version (just specify e.g. "v2" at the end of the ID, in the usual arXiv notation).
      - Duplicates are not downloaded.
        - The tool checks the specified output directory (default: current working directory) whether there is already a PDF file with a matching (versioned) arXiv ID in the filename.
      - The tool automatically respects the one-request-per-three-seconds-of-wall-time limit of the arXiv API TOS.
    - For instructions, see the [visualizer user manual](raven/visualizer/README.md).

  - New command-line tool: *Raven-burstbib*.
    - This takes a BibTex `.bib` file, and splits it into many `.bib` files, each containing one individual entry from the input file.
      - The files are automatically named based on the slug (BibTeX item ID), omitting any characters that are not valid in a filename.
      - If the output file already exists, the tool appends "_2", "_3", ... to the filename until it finds a filename that is not in use.
    - This is convenient to turn a huge BibTeX file (full of scientific abstracts) into individual documents for *Raven-librarian*'s document database, so that you can synthesize information over them with your LLM.
      - Run `raven-burstbib -o my_topic my_references.bib` and then copy/move the `my_topic` subdirectory to `~/.config/raven/llmclient/documents/`.
      - Raven-librarian will pick them up at next start, or immediately (if already running).
        - Note that Librarian's search indexing may take a while. For progress messages, see the terminal window from which you started *Raven-librarian*.
    - The current implementation of `raven-burstbib` is hacky. It doesn't actually parse the file properly, but only splits at BibTeX item headers.
      - It will handle invalid slugs properly, but other types of invalid input may cause crashes or unexpected behavior. It does work if your input file is valid BibTeX. :)
      - If you encounter a bug, please [open an issue](https://github.com/Technologicat/raven/issues).

  - New command-line tool: *Raven-dehyphenate*.
    - This uses the `dehyphen` Python package to sanitize text bro-ken by hyp-he-na-tion.
    - This is useful for `pdftotext` outputs, and for text obtained from PDF files by OCR (such as with `ocrmypdf --force-ocr input.pdf output.pdf`).
    - Raven-server's `sanitize` module is used automatically, if the server is reachable and the module is loaded on the server; else the dehyphenator model is loaded locally.

- *Raven-visualizer*:
  - Importer: New keyword detection mode "llm".
    - This uses the LLM backend configured for *Librarian*.
      - When this mode is used, the LLM backend must be running when *Visualizer* (or the command-line tool `raven-importer`) is started.
    - To initialize the task, this uses the same system prompt and AI character as *Librarian* uses for its chat.
      - This gives results consistent with what *Librarian* would say, because the LLM operations are handled by the same AI simulacrum.
      - See `raven.librarian.config`.
    - The AI analyzes the titles and abstracts for each cluster (separately), and suggests keywords. These keywords are recorded as the cluster keywords for *Visualizer*.

- *Raven-librarian*:
  - New feature: STT (speech to text, speech recognition). Talk to the AI using your mic!
    - To start speaking to the AI, click the mic button next to the chat text entry field (hotkey Ctrl+Shift+Enter).
      - The mic starts glowing red, to indicate that Librarian is listening. The VU meter (audio input level) next to the mic button becomes active.
      - To stop speaking, and send the spoken message to the AI, click the mic button again, or wait until the recorder detects silence and stops automatically.
        - The gray line on the VU meter is the silence threshold level.
      - The mic stops glowing (and returns to its default white).
    - Librarian then runs the recorded audio through a locally hosted [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) speech recognizer, which lives in the `stt` module of Raven-server.
    - The transcribed text is then sent to the AI, just as if it was typed as text to the chat text entry field.
    - For now, this feature has certain limitations:
      - The recorder autostop settings are hardcoded: 1.5s of input audio signal level under -40.00 dBFS.
        - This should work under most circumstances, but if you are not in "most circumstances", you'll have to stop recording by clicking the mic button again.
      - It is not possible to edit the transcribed text before sending.
    - To choose your mic device, see `raven.client.config`.
      - By default, Librarian picks the first available NON-monitoring audio capture device, in the order listed by the command-line tool `raven-check-audio-devices`.
      - The default should work on laptops, and in general, most systems that have just one audio input device.
      - The help card (F1) shows which mic device is active. It is also printed to the client log at app startup.
  - Very rudimentary chat branch navigation added.
    - In the linearized chat view, each message has buttons for next/previous sibling, jump 10 siblings, jump to first/last sibling.
      - Hotkeys apply to the last message displayed in the view.
    - It's easier to show than explain (try it out yourself!); but when switching siblings in the linearized chat view:
      - The sibling node switched to becomes the candidate HEAD.
      - If the candidate HEAD has any child nodes (i.e. chat continuations):
        - The child node with the most recent payload (according to payload timestamp) is chosen.
        - That child node becomes the new candidate HEAD, and the process repeats.
      - Once no more child nodes are found (i.e. the candidate HEAD is a leaf node), the candidate HEAD becomes the final new HEAD.
      - The linearized chat view scrolls to the sibling node that was switched to, regardless of where the final HEAD is.
    - Contrast this with the branch button, which sets the chat HEAD to the given node, without scanning the subtree for continuations.
      - Just like in `git`, branching is cheap. Branching only sets the HEAD pointer; no data is copied.
      - If you branch, but then change your mind, click the "Show chat continuation" button on the last message (hotkey Ctrl+Down).
        This rescans the chat continuation just like when switching siblings.
  - The LLM system prompt, the AI's character card, persona names (AI and user), and the AI's greeting message can now be customized in `raven.librarian.config`.
    - Changes take effect when Librarian is restarted.
    - Limitation: for now, only one AI character icon is loaded. If you switch characters, old chats will show the current character's icon (the persona name is stored in the chat database, but the avatar and icon paths are not).

**Changed**:

- *Raven-visualizer*:
  - Configurable plotter colors (background, grid, colormap). Loaded from `raven.visualizer.config` at app startup.
  - Configurable word cloud colors (background, colormap). Loaded from `raven.visualizer.config` at app startup.
  - The section headings in the BibTeX import dialog are now clickable, and perform the same function as the icon buttons.
  - *Visualizer*'s importer now automatically uses *Raven-server* for embeddings and NLP if it is running.
    - If the server is not running, the AI models are loaded locally (in the client process) as before.
    - There is no visible difference from the user's perspective (other than saving some VRAM, if also *Librarian* is running at the same time).
  - The importer now sanitizes abstracts using the `sanitize` module of Raven-server. This feature is on by default.
    - This affects only new BibTeX imports. Existing datasets are not modified.
    - The feature can be turned off in `raven.visualizer.config`. See the `dehyphenate` setting.
    - For each abstract, all paragraphs are sent together for processing. This may cause paragraphs to run together, if an abstract contains multiple paragraphs,
      but is often the only way if the input text is REALLY broken and contains newlines at arbitrary places. It was felt this is preferable, because scientific
      abstracts are often just one 200-word paragraph.
    - Raven-server's `sanitize` module is used automatically, if the server is reachable and the module is loaded on the server; else the dehyphenator model is loaded locally.

- *Raven-librarian*:
  - The document database now uses *Raven-server* for embeddings and NLP.
    - This saves some VRAM, by avoiding loading another copy of the same models in the client process.
    - This also makes the `raven.librarian.hybridir` information retrieval backend fully client-server, allowing the AI components for this too to run on another machine.
    - Because *Librarian* requires *Raven-server* for other purposes, too, *Librarian* will not start if the server is not running.
  - The document database now ingests `.bib` files, too.
    - This allows using the `raven-burstbib` command-line tool to mass-feed abstracts into Librarian's document database.
      - The tool takes a `.bib` file and splits it into individual files, one per entry. Hence each entry becomes a separate document in Librarian's document database.
  - The app now recovers if `state.json` is missing or corrupt.
  - Many small UI improvements, for example:
    - Window resizing implemented.
    - Collapsible thinking traces.
    - Interrupt/continue.
    - Avatar idle off.
      - Configurable, optional. See `avatar_config.idle_off_timeout` in `raven.librarian.config`. Seconds as float, or `None` to disable.
      - This saves some GPU compute by switching off the avatar video after the AI avatar is idle for a while.
      - The avatar video switches back on when:
        - The AI starts processing (writing new message, continuing existing message, rerolling existing message).
        - The chat view is re-rendered (e.g. by switching chat branches, or resizing the window).
        - The AI starts speaking (Ctrl+S, send last message to TTS).
    - Help card added.
    - TTS audio playback device setup in `raven.client.config`:
      - For configuration symmetry reasons, `None` now means "use the first available playback device as listed by `raven-check-audio-devices`", not the system's default playback device.
      - The new special value "system-default" uses the system's default playback device, so that the playback goes to the same device as from other apps. (This is what `None` did before.)
      - The default configuration has been changed to use "system-default", so that behavior should remain the same.

- Tools:
  - *Raven-pdf2bib*: Overhauled. See updated instructions in the [visualizer user manual](raven/visualizer/README.md).
    - To initialize each LLM task, this uses the same system prompt and AI character as *Librarian* uses for its chat.
      - This gives results consistent with what *Librarian* would say, because the LLM operations are handled by the same AI simulacrum.
      - See `raven.librarian.config`.


**Fixed**:

- *Raven-visualizer*:
  - Fix bug: "reset zoom" missed some datapoints (in a "select visible", hotkey F9), if they were exactly at the edges of the data bounding box.
    - Note that also loading a dataset resets the zoom, so the bug also affected the initial view upon loading a dataset.
    - Workaround for previous versions: after a "reset zoom", zoom out by one mouse wheel click before using "select visible".
  - Fix bug: wrong dtype in the embedder loader's CPU fallback.
    - The CPU fallback loader now always uses float32.
    - Workaround for previous versions: when working without a GPU, configure the embedder explicitly to use dtype `torch.float32`. See `raven.visualizer.config` and `raven.server.config`.
  - Fix UI bug: the plotter axes no longer light up when the mouse hovers on them.
    - The axes are not clickable, so the highlight was spurious.
    - This was broken when we upgraded to DearPyGUI 2.0, where the plotter changed to introduce that hover-highlight by default. Now we disable the highlight by theming the plotter.

- *Raven-avatar*:
  - Fix bug: Also the background image is now hidden while the avatar is paused.

- *Raven-librarian*:
  - Fix bug: The avatar's subtitle now re-positions itself correctly when the GUI is resized while the avatar is speaking.


---

**0.2.3** (7 October 2025):

**Added**:

- Prototype of *Raven-librarian*, a scientific LLM frontend GUI app.
  - Features an animated AI avatar with TTS and auto-translated subtitles, document database (plain text files for now), and tool-calling support (websearch for now).
    - Document database uses hybrid search (BM25 for keyword search, ChromaDB for semantic search).
  - In this prototype, chats are saved, but going back to previous chats is not yet possible because the GUI for that has not yet been developed.
  - When the *Documents* checkbox in the GUI is ON, the document database is autosearched, using the user's latest message to the AI as the query.
    - If, additionally, the *Speculation* checkbox is OFF, the LLM is bypassed when there is no match in the document database.
  - Websearch is enabled when the *Tools* checkbox in the GUI is ON.
  - Requires both *Raven-server* and the LLM backend (oobabooga/text-generation-webui, with `--api`) to be running.
  - For configuring Raven-librarian, for now, see `raven.librarian.config`.
    - The default location for the document database is `~/.config/raven/llmclient/documents`.
      - Librarian monitors this directory automatically, and also scans for offline changes at app startup.
      - Put `.txt` files there; they are search-indexed automatically. Replace files; the index is updated automatically. Remove files; they are removed from the index automatically.
      - If you need to force a manual index rebuild: make sure Librarian is not running, then delete `~/.config/raven/llmclient/rag_index`. It will be rebuilt at app startup.

- *Raven-avatar* now has a "data eyes" effect, for use as an LLM tool access indicator in Librarian.
  - Cel animation, up to 4 frames. Can be tested in `raven-avatar-settings-editor`.

- Speech video recording in `raven-avatar-settings-editor`.
  - Output goes in the `rec/` subdirectory.
  - TTS speech is saved as `.mp3` files, one per sentence.
  - Avatar video (avatar only, no background) is saved as as individual frames as `.qoi`.
    - For converting the video frames into a usable format, see the `raven-qoi2png` tool.
  - A speech timings list is saved as `.txt`.
  - These can be used to piece together a speech video in a video editor such as *OpenShot*.


**Changed**:

- *Raven-visualizer*'s importer now uses both the title and the abstract to cluster the inputs.
  - This requires *Snowflake-Arctic* or better as the embedding model; the older *mpnet* model tends to lead everything to become one cluster if the abstracts are used for clustering.
  - Old datasets must be imported again for the changes to take effect, because the embeddings are computed at import time.
    - Old embeddings caches must be deleted before re-importing!
    - For example, when `mydata.bib` is imported, Raven's importer produces:
      - `mydata_embeddings_cache.npz`: The vector embeddings. Delete this cache file!
      - `mydata_nlp_cache.pickle`: The natural language processing results, used for keyword detection. This cache is not affected by this change, so no need to delete.

- *Raven-server* now hosts all AI components, including embeddings and spaCy NLP.
  - spaCy NLP is only available for Python-based clients running the same versions of Python and spaCy, because it communicates in spaCy's internal format.


---

**0.2.2** (13 August 2025):

**Added**:

- First complete tech demo of *Raven-avatar*.
  - See the GUI apps `raven-avatar-settings-editor` (completely new postprocessor settings GUI) and `raven-avatar-pose-editor` (ported from the old THA3 pose editor).
  - The settings editor requires *Raven-server* to be running.

- *Raven-avatar* now has cel-blending and animefx support.


---

**0.2.1** (18 June 2025):

Otherwise the same as 0.2.0 (17 June 2025), but with the TODO cleaned up. Documenting both here.

**Added**:

- *Raven-server*: to provide an animated AI avatar, and to eventually host all AI components.
  - This is a web API server, initially ported and stripped from the discontinued *SillyTavern-extras*.
    - AGPL license! Affects the `raven.server` and `raven.avatar.pose_editor` directories.
    - All other components of *Raven* remain BSD-licensed. This includes `raven.avatar.settings_editor`.
  - Pose editor ported from wxPython to DPG, to match the rest of the *Raven* constellation, and to require only one GUI toolkit.
  - Added in d42d52356d61d290d4d9a1e5ffd0e1b6e0843c61, 22 May 2025. The entrypoint has since moved to `raven.server.app`.
  - Avatar has new features compared to the old Talkinghead:
    - Lipsync, to new TTS module based on Kokoro-82M.
    - Anime4k upscaling (super-resolution).


---


**0.1.x** and older

The project was started in December 2024.

No changelog was maintained.

These versions included only *Raven-visualizer*.
