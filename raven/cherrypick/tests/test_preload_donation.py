"""Tests for `raven.cherrypick.preload.donate_outgoing_image`.

The donation takes pixels out of the image viewer and files them in a cache under a key. Getting that key
wrong does not fail, crash, or log: the cache simply serves one image's pixels whenever another image is
asked for, and keeps doing it until something reloads that entry from disk. It survives leaving compare
mode, and it survives leaving the folder view — so a single wrong key is visible for the rest of the
session, at a distance from anything that looks related.

That is what happened on 2026-08-14. The donation was keyed by an index the app tracked as "the image
currently shown", which compare mode invalidated by loading frames straight into the viewer without going
through the app's own loader. Cancelling compare mode then donated the frame it happened to be parked on
under the index from before compare mode started. The user-visible symptom was compare mode appearing to
skip one image every loop — that image's key now held a different image's pixels — and it took a day to
find, because nothing near the corruption was wrong.

These tests use fakes rather than a DPG context: what is under test is which key a payload is filed under,
which is pure bookkeeping. `ImageView`'s side of the contract — that `take_mip_arrays` returns the key that
came in with the pixels — is expressed by the fake, so the two halves are pinned as a pair.
"""

import pytest

from raven.cherrypick.preload import donate_outgoing_image


class FakeImageView:
    """The part of `ImageView`'s contract the donation depends on.

    Carries an opaque `image_key` alongside the pixels, exactly as `ImageView` does, and hands both back
    together. `payload` stands in for the mip arrays and is unique per image, so a test can tell which
    image's data ended up where.
    """

    def __init__(self):
        self.mip_loading = False
        self._payload = None
        self._img_w = 0
        self._img_h = 0
        self._image_key = None

    def load(self, payload, img_w, img_h, *, image_key):
        self._payload = payload
        self._img_w = img_w
        self._img_h = img_h
        self._image_key = image_key

    def take_mip_arrays(self):
        if self._payload is None:
            return None
        payload, self._payload = self._payload, None
        return payload, self._img_w, self._img_h, self._image_key


class FakePreloadCache:
    """Records donations as ``{key: (payload, w, h)}``, and every call in order."""

    def __init__(self):
        self.entries = {}
        self.calls = []

    def donate(self, idx, mips, img_w, img_h):
        self.entries[idx] = (mips, img_w, img_h)
        self.calls.append(idx)


def test_donation_is_keyed_by_what_the_viewer_holds():
    iv, preload = FakeImageView(), FakePreloadCache()
    iv.load("pixels-of-7", 1344, 768, image_key=7)

    donate_outgoing_image(iv, preload)

    assert preload.entries == {7: ("pixels-of-7", 1344, 768)}


def test_a_second_loader_does_not_misfile_the_donation():
    """The compare-mode shape: someone else loads into the viewer, then the app donates.

    The app has no say in which image is on hand at that moment, so the key must come from the viewer.
    """
    iv, preload = FakeImageView(), FakePreloadCache()
    iv.load("pixels-of-15", 1152, 896, image_key=15)  # the app loaded this one...
    iv.load("pixels-of-20", 1344, 768, image_key=20)  # ...compare mode then loaded this one

    donate_outgoing_image(iv, preload)

    assert preload.entries == {20: ("pixels-of-20", 1344, 768)}
    assert 15 not in preload.entries, "image 15's key must not receive image 20's pixels"


def test_nothing_is_donated_while_mips_are_still_loading():
    """A partial mip set displays wrongly when taken again, so an incomplete image is not worth caching."""
    iv, preload = FakeImageView(), FakePreloadCache()
    iv.load("pixels-of-3", 640, 480, image_key=3)
    iv.mip_loading = True

    donate_outgoing_image(iv, preload)

    assert preload.calls == []


def test_an_unidentified_image_is_not_donated():
    """Without a key there is no safe place to file the pixels, and any guess would corrupt the cache."""
    iv, preload = FakeImageView(), FakePreloadCache()
    iv.load("pixels-of-nowhere", 640, 480, image_key=None)

    donate_outgoing_image(iv, preload)

    assert preload.calls == []


def test_an_empty_viewer_donates_nothing():
    iv, preload = FakeImageView(), FakePreloadCache()

    donate_outgoing_image(iv, preload)

    assert preload.calls == []


@pytest.mark.parametrize("iv, preload", [(None, FakePreloadCache()),
                                         (FakeImageView(), None),
                                         (None, None)])
def test_a_missing_component_is_not_an_error(iv, preload):
    """Both are `None` before a folder is opened, and the app calls this on every navigation."""
    donate_outgoing_image(iv, preload)
