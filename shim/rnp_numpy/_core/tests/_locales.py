"""A stand-in for upstream's `numpy._core.tests._locales` helper.

Same behaviour: find a locale whose decimal point is a comma and run the
test class under it, skipping when none is available.
"""

import locale
import sys

import pytest


def find_comma_decimal_point_locale():
    if sys.platform == "win32":
        locales = ["FRENCH"]
    else:
        locales = ["fr_FR", "fr_FR.UTF-8", "fi_FI", "fi_FI.UTF-8"]
    old = locale.setlocale(locale.LC_NUMERIC)
    try:
        for loc in locales:
            try:
                locale.setlocale(locale.LC_NUMERIC, loc)
            except locale.Error:
                continue
            if locale.localeconv()["decimal_point"] == ",":
                return old, loc
    finally:
        locale.setlocale(locale.LC_NUMERIC, locale=old)
    return old, None


class CommaDecimalPointLocale:
    (cur_locale, tst_locale) = find_comma_decimal_point_locale()

    def setup_method(self):
        if self.tst_locale is None:
            pytest.skip("No French locale available")
        locale.setlocale(locale.LC_NUMERIC, locale=self.tst_locale)

    def teardown_method(self):
        locale.setlocale(locale.LC_NUMERIC, locale=self.cur_locale)

    def __enter__(self):
        if self.tst_locale is None:
            pytest.skip("No French locale available")
        locale.setlocale(locale.LC_NUMERIC, locale=self.tst_locale)

    def __exit__(self, type, value, traceback):
        locale.setlocale(locale.LC_NUMERIC, locale=self.cur_locale)
