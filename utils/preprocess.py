import unicodedata


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_non_segment(char):
    """ If a char is not included in segmentation result. """
    cp = ord(char)
    return cp == 0 or cp == 0xfffd or _is_control(char) or \
            _is_whitespace(char)


def get_segment_char_span(example):
    sentence = example.sentence
    segment_char_span = []
    char_start = 0
    while char_start < len(sentence):
        segment = example.segments[len(segment_char_span)]
        # remove additional whitespaces from segment result
        segment = "".join(char for char in segment if not _is_whitespace(char))
        matched = False
        for char_end in range(char_start, len(sentence)):
            # remove whitespaces and special chars from sentence
            sentence_span = "".join(char for char in sentence[char_start:char_end + 1]
                                    if not _is_non_segment(char))
            if sentence_span == segment:
                # assign whitespaces and special chars to previous segments
                while char_end + 1 < len(sentence) and _is_non_segment(sentence[char_end + 1]):
                    char_end += 1
                matched = True
                segment_char_span.append([char_start, char_end])
                char_start = char_end + 1
                break
        assert matched, (sentence, segment)
