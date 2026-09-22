# The file dialog

Every Raven app that opens or saves a file uses the same file browser — *Raven-visualizer*'s **Open
dataset**, *Raven-librarian*'s attach button, *Raven-cherrypick*'s **Open image folder**, the two avatar
editors, and *Raven-xdot-viewer*. So it is documented once, here, rather than in each app's manual.

It can be driven entirely from the keyboard. If you would rather click with the mouse, everything below
has a control to click, and the keys are an alternative rather than a requirement.

**Contents:**

- [Finding a file by typing](#finding-a-file-by-typing)
- [Moving around](#moving-around)
- [Going somewhere else entirely](#going-somewhere-else-entirely)
- [Choosing what the listing shows](#choosing-what-the-listing-shows)
- [Picking more than one file](#picking-more-than-one-file)
- [Where the keyboard is](#where-the-keyboard-is)
- [Keyboard reference](#keyboard-reference)

## Finding a file by typing

**Just type.** The text field filters the listing as you go, on the same rule as *Raven-visualizer*'s
search: every fragment you type must occur somewhere in the name, in any order. `cat photo` finds
*photocatalytic*, and `2024 rep` finds *annual-report-2024.pdf*.

**The field's own text says whether anything matched** — green while something in the folder does, red
once nothing does. So a listing that has gone empty is distinguishable at a glance from a typo. Typing
`..` counts as a match: the way up answers a search like any other name.

In a **save** dialog the field names the file to be written rather than searching for one, so it is left
uncoloured — a name that matches nothing is the ordinary case there.

**`Tab` carries the name across.** From the field it moves the caret into the listing, completing what
you typed; from the listing it brings the caret back, carrying the name of whatever the cursor is on. So
the two halves of "type a bit, look at what is left, take that one" are one key. `Ctrl+F` returns the
caret to the field without completing, keeping what you typed.

## Moving around

**`Enter` goes as deep as the entry allows** — into the folder under the cursor, or accepting the file
under it when there is nothing deeper. **`Ctrl+Enter` accepts where you are** without going deeper,
which is the same action as the **OK** button.

That distinction is the one worth learning, because it is what lets a folder picker work at all: walk
into a folder with `Enter`, and take the folder you are standing in with `Ctrl+Enter`. A line above the
buttons always names the path **OK** would return, and updates as you move, so you can see which it
would be before committing.

The arrow keys, `Page Up` / `Page Down` and `Home` / `End` move the cursor through the listing.
`Left` / `Right` step one entry back and forward — which matters in the thumbnail view, whose rows hold
several tiles each.

`Alt+Up` goes up one level, and `Ctrl+Up` does the same one-handed. `Ctrl+Home` returns to the folder the
dialog started in, and `F5` re-reads the current one from disk (useful e.g. if you created or copied
a file there in another app while the dialog was open).

## Going somewhere else entirely

Two routes out of wherever you are:

**The shortcuts panel** down the left side lists your places and drives. `Ctrl+B` hands it the arrow
keys; `Up` / `Down`, `Page Up` / `Page Down` and `Home` / `End` move through it, and a letter key jumps
to the next entry starting with that letter — pressing it again cycles, and `Shift` with it goes
backwards. `Enter` goes to the highlighted place, and `Esc` gives the keyboard back to the find field.

**The path field** is for the paths that do not come from browsing: one pasted from a terminal,
from an OS file browser, or from a message, or a short root like `/mnt` that is nowhere near where
you are and in nobody's shortcuts. `Ctrl+L` puts the caret in it.

**It says what `Enter` will do with it** as you type — green while it names a folder that exists, red
once it cannot lead anywhere, and plain while you are on your way to one (partial match). So a path
that is stale, or mistyped at the far end, shows it as you go rather than as a message box after you
commit. Once `Enter` has taken you there the field goes plain again: the colours belong to the typing.
A `~` is judged by the folder it stands for, while the field keeps showing what you typed.

It does not complete with `Tab`, and does not need to — the find field does that better, a fragment at a
time in any order.

## Choosing what the listing shows

**File types.** Where the app asked for particular types, `Ctrl+1` … `Ctrl+9` pick the Nth offered type,
and `Ctrl+Shift+F` hands the arrow keys to the type list so you can step through them and watch the
listing narrow. `Esc` gives the keyboard back to the find field.

A dialog whose caller named no types offers *all files* and nothing else, and **a folder picker offers no
type filter at all** — choosing among folders is not something a type filter can narrow, since it applies
to files and would hide the folders you navigate through to reach them.

**Sorting.** `Ctrl+Shift+1` … `Ctrl+Shift+4` sort by name, date, type or size. Pressing the same one
again reverses it.

**Hidden files.** `Ctrl+H`, or the *Hidden* checkbox, shows or hides dotfiles and hidden folders. The
choice holds until you change it back.

**Thumbnails.** Where the dialog lists files at all, `Ctrl+T` switches between the list and a grid of
image previews. In a folder picker the pictures are shown but dimmed and unclickable: they are there to
tell you whether this is the right folder, not to be chosen.

## Picking more than one file

Where the dialog was opened for several files, `Ctrl+Space` marks or unmarks the entry under the cursor —
what `Ctrl+click` does with the mouse. The line above the buttons updates as you mark, including when you
mark a folder.

## Where the keyboard is

**A blue border marks whichever control has the arrow keys** — the find field, the path field, the file
type list, the listing, or the shortcuts panel — so a chord that hands the keys elsewhere shows where
they went.

**A blue cursor, breathing slowly, marks the entry the keyboard is on.** The two can be lit at once on
purpose: the border says where the keys are, and the cursor says what `Enter` would act on, which in this
dialog are different questions.

`Esc` cancels the dialog — or, when the keyboard is in a side control, hands it back to the find field
first.

## Keyboard reference

`F1` inside the dialog shows this list on a card, and offers only the keys *that* dialog can actually
use: marking appears only where several files may be picked, thumbnails only where there are files to
show, and the type-filter keys only where types were offered.

| Key | What it does |
|---|---|
| `Up` / `Down` | Move the cursor one row |
| `Page Up` / `Page Down` | Move about a screenful |
| `Home` / `End` | First / last entry |
| `Left` / `Right` | Previous / next entry, once `Tab` has put the caret in the listing |
| `Enter` | Go as deep as this entry allows — into a folder, or accept a file |
| `Ctrl+Enter` | Accept without going deeper (the **OK** button) |
| `Esc` | Cancel, or out of a side control |
| `Ctrl+Space` | Mark or unmark this entry, where several may be picked |
| `Alt+Up` | Up one level |
| `Ctrl+Up` | The same, one-handed |
| `Ctrl+Home` | Back to the starting folder |
| `F5` | Re-read this folder |
| Type anything | Find in this folder — fragments, in any order. In a save dialog, name the file to save as |
| `Tab` | Caret to the listing, completing what you typed; again brings it back carrying the cursor's name |
| `Ctrl+F` | Caret back to the field, keeping what you typed |
| `Ctrl+L` | Caret to the path field |
| `Ctrl+B` | Caret to the shortcuts panel |
| `A` … `Z` | In the shortcuts panel, jump by first letter; again cycles, `Shift` reverses |
| `Ctrl+1` … `Ctrl+9` | Show the Nth file type |
| `Ctrl+Shift+F` | Browse the file types with the arrow keys |
| `Ctrl+Shift+1` … `Ctrl+Shift+4` | Sort by name / date / type / size; again to reverse |
| `Ctrl+H` | Show or hide hidden files |
| `Ctrl+T` | Thumbnails, or the list |
| `F1` | Open the help card |
