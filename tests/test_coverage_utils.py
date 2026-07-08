"""Tests to improve coverage for _utils.py and _compat.py."""

import warnings

import numpy as np
import pandas as pd
import pytest

from ggplot2_py._compat import (
    cli_abort,
    cli_warn,
    cli_inform,
    is_string,
    is_bool,
    is_character,
    is_null,
    is_bare_list,
    is_true,
    is_false,
    is_scalar_character,
    is_scalar_logical,
    is_installed,
    check_installed,
    deprecate_warn,
    deprecate_soft,
    deprecate_stop,
    Waiver,
    waiver,
    is_waiver,
    caller_arg,
)
from ggplot2_py._utils import (
    remove_missing,
    resolution,
    snake_class,
    has_groups,
    empty,
    is_empty,
    try_fetch,
    compact,
    modify_list,
    data_frame,
    unique_default,
    rename,
    id_var,
    plyr_id,
    interleave,
    width_cm,
    height_cm,
    stapled_to_list,
)


# =====================================================================
# _compat.py tests
# =====================================================================

class TestCliAbort:
    def test_raises_value_error_default(self):
        with pytest.raises(ValueError, match="test error"):
            cli_abort("test error")

    def test_raises_custom_exception(self):
        with pytest.raises(TypeError, match="type err"):
            cli_abort("type err", cls=TypeError)

    def test_format_kwargs(self):
        with pytest.raises(ValueError, match="hello world"):
            cli_abort("hello {name}", name="world")

    def test_bad_format_fallback(self):
        with pytest.raises(ValueError, match=r"\{missing\}"):
            cli_abort("{missing}")

    def test_call_parameter(self):
        with pytest.raises(ValueError):
            cli_abort("msg", call="some_fn")


class TestCliWarn:
    def test_emits_warning(self):
        with pytest.warns(UserWarning, match="watch out"):
            cli_warn("watch out")

    def test_format_kwargs(self):
        with pytest.warns(UserWarning, match="value is 42"):
            cli_warn("value is {v}", v=42)

    def test_bad_format_fallback(self):
        with pytest.warns(UserWarning, match=r"\{x\}"):
            cli_warn("{x}")


class TestCliInform:
    def test_is_silent(self):
        # cli_inform is a no-op; just call it without error
        cli_inform("info message")
        cli_inform("info {x}", x=1)


class TestTypePredicates:
    def test_is_string(self):
        assert is_string("hello")
        assert not is_string(42)
        assert not is_string(["a"])

    def test_is_bool(self):
        assert is_bool(True)
        assert is_bool(False)
        assert not is_bool(1)
        assert not is_bool("true")

    def test_is_character(self):
        assert is_character("a")
        assert is_character(["a", "b"])
        assert not is_character([1, 2])
        assert not is_character(42)
        assert is_character([])  # empty list of strings

    def test_is_null(self):
        assert is_null(None)
        assert not is_null(0)
        assert not is_null("")

    def test_is_bare_list(self):
        assert is_bare_list([1, 2])
        assert not is_bare_list((1, 2))
        assert not is_bare_list("abc")

        class MyList(list):
            pass
        assert not is_bare_list(MyList())

    def test_is_true(self):
        assert is_true(True)
        assert not is_true(1)
        assert not is_true("true")

    def test_is_false(self):
        assert is_false(False)
        assert not is_false(0)
        assert not is_false("")

    def test_is_scalar_character(self):
        assert is_scalar_character("x")
        assert not is_scalar_character(42)

    def test_is_scalar_logical(self):
        assert is_scalar_logical(True)
        assert not is_scalar_logical(1)


class TestPackageChecks:
    def test_is_installed_existing(self):
        assert is_installed("os")
        assert is_installed("numpy")

    def test_is_installed_missing(self):
        assert not is_installed("nonexistent_pkg_12345")

    def test_check_installed_existing(self):
        check_installed("os")  # should not raise

    def test_check_installed_missing(self):
        with pytest.raises(ImportError, match="nonexistent_pkg"):
            check_installed("nonexistent_pkg")

    def test_check_installed_with_reason(self):
        with pytest.raises(ImportError, match="for plotting"):
            check_installed("nonexistent_pkg", reason="for plotting")


class TestDeprecation:
    def test_deprecate_warn(self):
        with pytest.warns(DeprecationWarning, match="was deprecated"):
            deprecate_warn("3.4.0", "qplot()")

    def test_deprecate_warn_with_replacement(self):
        with pytest.warns(DeprecationWarning, match="ggplot"):
            deprecate_warn("3.4.0", "qplot()", with_="ggplot()")

    def test_deprecate_soft(self):
        with pytest.warns(DeprecationWarning, match="was deprecated"):
            deprecate_soft("3.0.0", "some_fn()")

    def test_deprecate_stop(self):
        with pytest.raises(RuntimeError, match="defunct"):
            deprecate_stop("2.0.0", "old_fn()")

    def test_deprecate_stop_with_replacement(self):
        with pytest.raises(RuntimeError, match="new_fn"):
            deprecate_stop("2.0.0", "old_fn()", with_="new_fn()")


class TestWaiver:
    def test_singleton(self):
        w1 = Waiver()
        w2 = Waiver()
        assert w1 is w2

    def test_repr(self):
        assert repr(waiver()) == "waiver()"

    def test_bool_is_false(self):
        assert not waiver()
        assert bool(waiver()) is False

    def test_is_waiver(self):
        assert is_waiver(waiver())
        assert not is_waiver(None)
        assert not is_waiver(0)

    def test_waiver_func_returns_instance(self):
        assert isinstance(waiver(), Waiver)


class TestCallerArg:
    def test_returns_arg_name(self):
        assert caller_arg("x") == "x"
        assert caller_arg("data") == "data"


# =====================================================================
# _utils.py tests
# =====================================================================

class TestRemoveMissing:
    def test_empty_df(self):
        df = pd.DataFrame()
        result = remove_missing(df)
        assert result.empty

    def test_no_missing(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result = remove_missing(df, na_rm=True)
        assert len(result) == 3

    def test_with_missing_warns(self):
        df = pd.DataFrame({"x": [1, np.nan, 3], "y": [4, 5, 6]})
        with pytest.warns(UserWarning, match="Removed 1"):
            result = remove_missing(df)
        assert len(result) == 2

    def test_with_missing_na_rm_silent(self):
        df = pd.DataFrame({"x": [1, np.nan, 3]})
        result = remove_missing(df, na_rm=True)
        assert len(result) == 2

    def test_specific_vars(self):
        df = pd.DataFrame({"x": [1, np.nan, 3], "y": [np.nan, 5, 6]})
        result = remove_missing(df, vars=["x"], na_rm=True)
        assert len(result) == 2

    def test_missing_var_not_in_df(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = remove_missing(df, vars=["z"], na_rm=True)
        assert len(result) == 3

    def test_finite_removes_inf(self):
        df = pd.DataFrame({"x": [1, np.inf, 3], "y": [4, 5, 6]})
        result = remove_missing(df, finite=True, na_rm=True)
        assert len(result) == 2

    def test_finite_with_non_numeric(self):
        df = pd.DataFrame({"x": ["a", "b", None], "y": [1, 2, 3]})
        result = remove_missing(df, finite=True, na_rm=True)
        assert len(result) == 2

    def test_name_in_warning(self):
        df = pd.DataFrame({"x": [1, np.nan]})
        with pytest.warns(UserWarning, match="geom_point"):
            remove_missing(df, name="geom_point")

    def test_no_check_cols(self):
        df = pd.DataFrame({"x": [1, 2]})
        result = remove_missing(df, vars=[])
        assert len(result) == 2


class TestResolution:
    def test_basic(self):
        assert resolution([1, 2, 3]) == 1.0

    def test_single_value_with_zero(self):
        assert resolution([5]) == 1.0

    def test_single_value_without_zero(self):
        # R: zero_range(range(5)) → 1 regardless of `zero`
        assert resolution([5], zero=False) == 1.0

    def test_empty(self):
        assert resolution([]) == 1.0

    def test_with_nan(self):
        # R: resolution(c(1, NA, 3)) = 1 — zero=TRUE adds 0 to the
        # value set, so the smallest gap is 0→1
        assert resolution([1, np.nan, 3]) == 1.0
        assert resolution([1, np.nan, 3], zero=False) == 2.0

    def test_discrete(self):
        assert resolution([1, 2, 3], discrete=True) == 1.0

    def test_non_uniform(self):
        assert resolution([1, 2, 10]) == 1.0


class TestSnakeClass:
    def test_dataframe(self):
        assert snake_class(pd.DataFrame()) == "data_frame"

    def test_class_input(self):
        assert snake_class(str) == "str"

    def test_custom_class(self):
        class MyClassName:
            pass
        assert snake_class(MyClassName()) == "my_class_name"


class TestHasGroups:
    def test_no_group_column(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        assert not has_groups(df)

    def test_single_group(self):
        df = pd.DataFrame({"group": [1, 1, 1]})
        assert not has_groups(df)

    def test_multiple_groups(self):
        df = pd.DataFrame({"group": [1, 2, 1]})
        assert has_groups(df)


class TestEmpty:
    def test_empty_df(self):
        assert empty(pd.DataFrame())

    def test_non_empty_df(self):
        assert not empty(pd.DataFrame({"x": [1]}))

    def test_no_columns(self):
        df = pd.DataFrame(index=[0, 1, 2])
        assert empty(df)


class TestIsEmpty:
    def test_empty_df(self):
        assert is_empty(pd.DataFrame())

    def test_none(self):
        assert is_empty(None)

    def test_empty_list(self):
        assert is_empty([])

    def test_empty_dict(self):
        assert is_empty({})

    def test_non_empty(self):
        assert not is_empty([1, 2])

    def test_non_sized_object(self):
        assert not is_empty(42)


class TestDataFrame:
    def test_basic(self):
        df = data_frame(x=[1, 2], y=[3, 4])
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["x", "y"]


class TestUniqueDefault:
    def test_preserves_order(self):
        result = unique_default([3, 1, 2, 1, 3])
        np.testing.assert_array_equal(result, [3, 1, 2])


class TestIdVar:
    def test_basic(self):
        result = id_var(["a", "b", "a", "c"])
        assert result[0] == result[2]  # same group for "a"
        assert result[0] != result[1]
        assert min(result) == 1  # 1-based


class TestPlyrId:
    def test_basic(self):
        df = pd.DataFrame({"x": ["a", "b", "a"], "y": [1, 1, 2]})
        result = plyr_id(df)
        assert result[0] != result[2]  # ("a",1) != ("a",2)
        assert len(result) == 3

    def test_empty(self):
        df = pd.DataFrame()
        result = plyr_id(df)
        assert len(result) == 0

    def test_no_columns(self):
        df = pd.DataFrame(index=[0, 1])
        result = plyr_id(df)
        np.testing.assert_array_equal(result, [1, 1])


class TestRename:
    def test_basic(self):
        result = rename({"a": 1, "b": 2}, {"a": "x"})
        assert result == {"x": 1, "b": 2}

    def test_kwargs(self):
        result = rename({"a": 1}, b="y")
        assert result == {"a": 1}

    def test_no_mapping(self):
        result = rename({"a": 1})
        assert result == {"a": 1}


class TestTryFetch:
    def test_callable_success(self):
        assert try_fetch(lambda: 42) == 42

    def test_callable_failure(self):
        assert try_fetch(lambda: 1 / 0, default="oops") == "oops"

    def test_non_callable(self):
        assert try_fetch(42) == 42


class TestCompact:
    def test_removes_none(self):
        result = compact({"a": 1, "b": None, "c": 3})
        assert result == {"a": 1, "c": 3}

    def test_keeps_falsy_non_none(self):
        result = compact({"a": 0, "b": "", "c": False})
        assert len(result) == 3


class TestModifyList:
    def test_basic_override(self):
        result = modify_list({"a": 1, "b": 2}, {"b": 3, "c": 4})
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_none_removes(self):
        result = modify_list({"a": 1, "b": 2}, {"b": None})
        assert result == {"a": 1}


class TestInterleave:
    def test_basic(self):
        assert interleave([1, 2, 3], [10, 20, 30]) == [1, 10, 2, 20, 3, 30]

    def test_empty(self):
        assert interleave() == []

    def test_uneven_lengths(self):
        result = interleave([1, 2], [10, 20, 30])
        assert result == [1, 10, 2, 20, 30]


class TestWidthCm:
    def test_scalar(self):
        assert width_cm(5.0) == 5.0

    def test_array(self):
        result = width_cm([1.0, 2.0])
        np.testing.assert_array_equal(result, [1.0, 2.0])


class TestHeightCm:
    def test_scalar(self):
        assert height_cm(3.0) == 3.0

    def test_array(self):
        result = height_cm([1.0, 2.0])
        np.testing.assert_array_equal(result, [1.0, 2.0])


class TestStapledToList:
    def test_list(self):
        assert stapled_to_list([1, 2]) == [1, 2]

    def test_none(self):
        assert stapled_to_list(None) == []

    def test_scalar(self):
        assert stapled_to_list(42) == [42]
