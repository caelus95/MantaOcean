# Copy this to matplotlib.cbook __init__.py 521th line


# @deprecated("3.1", alternative="inspect.cleandoc")
def dedent(s):
    """
    Remove excess indentation from docstring *s*.

    Discards any leading blank lines, then removes up to n whitespace
    characters from each line, where n is the number of leading
    whitespace characters in the first line. It differs from
    textwrap.dedent in its deletion of leading blank lines and its use
    of the first non-blank line to determine the indentation.

    It is also faster in most cases.
    """
    # This implementation has a somewhat obtuse use of regular
    # expressions.  However, this function accounted for almost 30% of
    # matplotlib startup time, so it is worthy of optimization at all
    # costs.

    if not s:      # includes case of s is None
        return ''

    match = _find_dedent_regex.match(s)
    if match is None:
        return s

    # This is the number of spaces to remove from the left-hand side.
    nshift = match.end(1) - match.start(1)
    if nshift == 0:
        return s

    # Get a regex that will remove *up to* nshift spaces from the
    # beginning of each line.  If it isn't in the cache, generate it.
    unindent = _dedent_regex.get(nshift, None)
    if unindent is None:
        unindent = re.compile("\n\r? {0,%d}" % nshift)
        _dedent_regex[nshift] = unindent

    result = unindent.sub("\n", s).strip()
    return result
