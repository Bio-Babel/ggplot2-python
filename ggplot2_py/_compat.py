"""
R infrastructure shims for ggplot2.

Replaces functionality from rlang, cli, lifecycle, withr, and vctrs that
ggplot2 relies on, adapted for idiomatic Python usage.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any, NoReturn, Optional

__all__ = [
    "cli_abort",
    "cli_warn",
    "cli_inform",
    "is_string",
    "is_bool",
    "is_character",
    "is_null",
    "is_bare_list",
    "is_true",
    "is_false",
    "is_scalar_character",
    "is_scalar_logical",
    "is_installed",
    "check_installed",
    "deprecate_warn",
    "deprecate_soft",
    "deprecate_stop",
    "Waiver",
    "is_waiver",
    "waiver",
    "caller_arg",
]


# ---------------------------------------------------------------------------
# CLI messaging helpers (rlang / cli replacements)
# ---------------------------------------------------------------------------

def cli_abort(
    message: str,
    *,
    call: Optional[str] = None,
    cls: type = ValueError,
    **kwargs: Any,
) -> NoReturn:
    """Raise an exception with a formatted message.

    Parameters
    ----------
    message : str
        Error message.  May contain ``{name}``-style placeholders that
        are filled from *kwargs*.
    call : str, optional
        Name of the calling function (for context in the message).
    cls : type, optional
        Exception class to raise.  Defaults to ``ValueError``.
    **kwargs : Any
        Substitution values for placeholders in *message*.

    Raises
    ------
    Exception
        An instance of *cls* with the formatted message.
    """
    try:
        formatted = message.format(**kwargs) if kwargs else message
    except (KeyError, IndexError):
        formatted = message
    raise cls(formatted)


def cli_warn(
    message: str,
    *,
    call: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Issue a ``UserWarning`` with a formatted message.

    Parameters
    ----------
    message : str
        Warning message.  May contain ``{name}``-style placeholders.
    call : str, optional
        Name of the calling function.
    **kwargs : Any
        Substitution values for placeholders in *message*.
    """
    try:
        formatted = message.format(**kwargs) if kwargs else message
    except (KeyError, IndexError):
        formatted = message
    warnings.warn(formatted, UserWarning, stacklevel=2)


def cli_inform(
    message: str,
    *,
    call: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Emit an informational message (no-op by default).

    In interactive sessions this could print; in batch mode it stays silent.
    Override by monkey-patching if verbose output is desired.

    Parameters
    ----------
    message : str
        Informational message.
    call : str, optional
        Name of the calling function.
    **kwargs : Any
        Substitution values for placeholders in *message*.
    """
    # Intentionally silent – mirrors rlang::inform() in non-interactive R.
    pass


# ---------------------------------------------------------------------------
# Type-checking predicates (rlang replacements)
# ---------------------------------------------------------------------------

def is_string(x: Any) -> bool:
    """Check whether *x* is a single string.

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    bool
        ``True`` if *x* is an instance of ``str``.
    """
    return isinstance(x, str)


def is_bool(x: Any) -> bool:
    """Check whether *x* is a Python ``bool``.

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    bool
        ``True`` if *x* is an instance of ``bool``.
    """
    return isinstance(x, bool)


def is_character(x: Any) -> bool:
    """Check whether *x* is a string or list of strings.

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    bool
        ``True`` if *x* is a ``str`` or a ``list`` whose elements are all
        strings.
    """
    if isinstance(x, str):
        return True
    if isinstance(x, list):
        return all(isinstance(el, str) for el in x)
    return False


def is_null(x: Any) -> bool:
    """Check whether *x* is ``None``.

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    bool
        ``True`` if *x* is ``None``.
    """
    return x is None


def is_bare_list(x: Any) -> bool:
    """Check whether *x* is a plain ``list`` (not a subclass).

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    bool
        ``True`` if ``type(x)`` is exactly ``list``.
    """
    return type(x) is list


def is_true(x: Any) -> bool:
    """Check whether *x* is literally ``True``.

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    bool
    """
    return x is True


def is_false(x: Any) -> bool:
    """Check whether *x* is literally ``False``.

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    bool
    """
    return x is False


def is_scalar_character(x: Any) -> bool:
    """Check whether *x* is a length-1 character value (a single string).

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    bool
    """
    return isinstance(x, str)


def is_scalar_logical(x: Any) -> bool:
    """Check whether *x* is a length-1 logical value (a single bool).

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    bool
    """
    return isinstance(x, bool)


# ---------------------------------------------------------------------------
# Package availability (rlang replacements)
# ---------------------------------------------------------------------------

def is_installed(pkg: str) -> bool:
    """Check whether a Python package is importable.

    Parameters
    ----------
    pkg : str
        Package name (dotted names are supported).

    Returns
    -------
    bool
        ``True`` if the package can be imported.
    """
    try:
        importlib.import_module(pkg)
        return True
    except ImportError:
        return False


def check_installed(
    pkg: str,
    reason: Optional[str] = None,
) -> None:
    """Raise ``ImportError`` if a package is not available.

    Parameters
    ----------
    pkg : str
        Package name.
    reason : str, optional
        Human-readable reason the package is needed (appended to the
        error message).

    Raises
    ------
    ImportError
        If the package cannot be imported.
    """
    if not is_installed(pkg):
        msg = f"The '{pkg}' package is required"
        if reason:
            msg += f" {reason}"
        msg += "."
        raise ImportError(msg)


# ---------------------------------------------------------------------------
# Deprecation helpers (lifecycle replacements)
# ---------------------------------------------------------------------------

def deprecate_warn(
    when: str,
    what: str,
    with_: Optional[str] = None,
) -> None:
    """Emit a deprecation warning.

    Parameters
    ----------
    when : str
        Version in which the feature was deprecated (e.g. ``"3.4.0"``).
    what : str
        Description of the deprecated feature.
    with_ : str, optional
        Replacement to suggest.
    """
    msg = f"{what} was deprecated in version {when}."
    if with_ is not None:
        msg += f" Please use {with_} instead."
    warnings.warn(msg, DeprecationWarning, stacklevel=2)


def deprecate_soft(
    when: str,
    what: str,
    with_: Optional[str] = None,
) -> None:
    """Emit a soft deprecation warning.

    Soft deprecations are shown only when the deprecated code is called
    from outside the package.  In this Python port the distinction is
    ignored and a normal ``DeprecationWarning`` is emitted.

    Parameters
    ----------
    when : str
        Version string.
    what : str
        Deprecated feature description.
    with_ : str, optional
        Replacement suggestion.
    """
    deprecate_warn(when, what, with_=with_)


def deprecate_stop(
    when: str,
    what: str,
    with_: Optional[str] = None,
) -> NoReturn:
    """Raise an error for a defunct (fully removed) feature.

    Parameters
    ----------
    when : str
        Version in which the feature was removed.
    what : str
        Description of the removed feature.
    with_ : str, optional
        Replacement to suggest.

    Raises
    ------
    RuntimeError
        Always raised.
    """
    msg = f"{what} was deprecated in version {when} and is now defunct."
    if with_ is not None:
        msg += f" Please use {with_} instead."
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Waiver sentinel (ggplot2-specific)
# ---------------------------------------------------------------------------

class Waiver:
    """Sentinel class indicating "use the default value".

    In R ggplot2, ``waiver()`` signals that a parameter should fall back to
    the default computed by the plot.  This Python equivalent works the
    same way.
    """

    _instance: Optional["Waiver"] = None

    def __new__(cls) -> "Waiver":
        # Singleton so that ``Waiver() is Waiver()`` holds.
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "waiver()"

    def __bool__(self) -> bool:
        # Prevent accidental truthiness tests.
        return False


def waiver() -> Waiver:
    """Return a ``Waiver`` sentinel value.

    Returns
    -------
    Waiver
        The singleton waiver instance.
    """
    return Waiver()


def is_waiver(x: Any) -> bool:
    """Check whether *x* is a ``Waiver`` sentinel.

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    bool
        ``True`` if *x* is a ``Waiver`` instance.
    """
    return isinstance(x, Waiver)


# ---------------------------------------------------------------------------
# Miscellaneous rlang helpers
# ---------------------------------------------------------------------------

def caller_arg(arg: str) -> str:
    """Return a human-readable label for a function argument.

    In R, ``rlang::caller_arg()`` inspects the call stack.  Here we
    simply return the argument name as-is.

    Parameters
    ----------
    arg : str
        Argument name.

    Returns
    -------
    str
        The same string, suitable for inclusion in error messages.
    """
    return arg
