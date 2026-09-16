import re

VALID_LEFT_TOKEN_LIST = [
    "\\left\\lbrace",
    "\\left\\lVert",
    "\\left\\lvert",
    "\\left\\rvert",
    "\\left\\rVert",
    "\\left\\vert",
    "\\left\\Vert",
    "\\left\\lfloor",
    "\\left\\lbrack",
    "\\left\\langle",
    "\\left|",
    "\\left\\|",
    "\\left[",
    "\\left]",
    "\\left(",
    "\\left)",
    "\\left\\{",
    "\\left\\}",
    "\\left.",
    "\\left/",
]

VALID_RIGHT_TOKEN_LIST = [
    "\\right\\rbrace",
    "\\right\\lVert",
    "\\right\\lvert",
    "\\right\\rvert",
    "\\right\\rVert",
    "\\right\\vert",
    "\\right\\Vert",
    "\\right\\rfloor",
    "\\right\\rbrack",
    "\\right\\rangle",
    "\\right|",
    "\\right\\|",
    "\\right]",
    "\\right[",
    "\\right)",
    "\\right(",
    "\\right\\}",
    "\\right\\{",
    "\\right.",
    "\\right/",
]

LEFT_TOKEN_LIST = [
    r"\\left\\lbrace",
    r"\\left\\lVert",
    r"\\left\\lvert",
    r"\\left\\rvert",
    r"\\left\\rVert",
    r"\\left\\vert",
    r"\\left\\Vert",
    r"\\left\\lfloor",
    r"\\left\\lbrack",
    r"\\left\\langle",
    r"\\left\|",
    r"\\left\\\|",
    r"\\left\[",
    r"\\left\]",
    r"\\left\(",
    r"\\left\\{",
    r"\\left\\}",
    r"\\left\.",
]


def count_left(latex):
    pattern = "|".join(LEFT_TOKEN_LIST)
    matches = re.findall(pattern, latex)
    return len(matches)


RIGHT_TOKEN_LIST = [
    r"\\right\\rbrace",
    r"\\right\\lVert",
    r"\\right\\lvert",
    r"\\right\\rvert",
    r"\\right\\rVert",
    r"\\right\\vert",
    r"\\right\\Vert",
    r"\\right\\rfloor",
    r"\\right\\rbrack",
    r"\\right\\rangle",
    r"\\right\|",
    r"\\right\\\|",
    r"\\right\]",
    r"\\right\[",
    r"\\right\)",
    r"\\right\\}",
    r"\\right\\{",
    r"\\right.",
]


def count_right(latex):
    pattern = "|".join(RIGHT_TOKEN_LIST)
    matches = re.findall(pattern, latex)
    return len(matches)


def check_left_right(latex: str) -> bool:
    return count_left(latex) == count_right(latex)


def check_align(latex: str) -> bool:
    return latex.count("\\begin{array}") == latex.count("\\end{array}")


def split_with_delimiters(s: str) -> list[str]:
    pattern = r"(\&|\\\\|\\begin{array}\s*{[a-zA-Z\s]*}|\\end{array})"
    return re.split(pattern, s)


def split_with_left_right(s: str) -> list[str]:
    pattern = "|".join(LEFT_TOKEN_LIST + RIGHT_TOKEN_LIST)
    return re.split(f"({pattern})", s)


def tag_array(node_list):
    node_stack = []
    array_list = []
    array_tag = 0
    for node_idx, node in enumerate(node_list):
        # array begin
        if "\\begin{array}" in node:
            array_tag += 1
            node_stack.append([array_tag, node_idx])
        # array end
        elif node == "\\end{array}":
            array_tag, node_idx_start = node_stack.pop()
            node_idx_end = node_idx
            array_list.append([array_tag, node_idx_start, node_idx_end])
        else:
            continue

    # find_contain
    for arr_idx1, arr1 in enumerate(array_list):
        contain = []
        for arr_idx2, arr2 in enumerate(array_list):
            if arr_idx1 == arr_idx2:
                continue
            if arr1[1] < arr2[1] and arr1[2] > arr2[2]:
                contain.append(array_list[arr_idx2][0])
        array_list[arr_idx1].append(contain)

    return array_list


def tag_element(node_list, array_list):
    if len(array_list) == 0:
        return [None]

    tag2arr = {arr[0]: arr for arr in array_list}

    node_tag_all_arr = []
    # from top to bottom
    for arr_idx in range(len(array_list)):

        contain_arr_tag = array_list[arr_idx][-1]
        node_list_cur_arr = [node for node in node_list]
        node_tag_cur_arr = [None for node in node_list]

        # mask contained array
        for arr_tag in contain_arr_tag:
            arr = tag2arr[arr_tag]
            for idx in range(arr[1], arr[2] + 1):
                node_list_cur_arr[idx] = None

        # mask not in the range
        for idx in range(len(node_list_cur_arr)):
            if idx < array_list[arr_idx][1] or idx > array_list[arr_idx][2]:
                node_list_cur_arr[idx] = None

        element_idx = 0
        for node_idx, node in enumerate(node_list_cur_arr):
            if node == "&" or node == "\\\\":
                element_idx += 1
            elif node is None:
                continue
            else:
                node_tag_cur_arr[node_idx] = (array_list[arr_idx][0], element_idx)

        node_tag_all_arr.append(node_tag_cur_arr)

    node_tag_list = [
        next(
            (item for lst in node_tag_all_arr if (item := lst[i]) is not None),
            None,
        )
        for i in range(len(node_tag_all_arr[0]))
    ]

    return node_tag_list


# fmt: off
def is_pair_left_right(token_l, token_r):
    if (token_l == "\\left\\lbrace" or token_l == "\\left.") and \
        (token_r == "\\right\\rbrace" or token_r == "\\right."):
        return True
    
    if (token_l == "\\left\\lVert" or token_l == "\\left.") and \
        (token_r == "\\right\\lVert" or token_r == "\\right."):
        return True
    
    if (token_l == "\\left\\lvert" or token_l == "\\left.") and \
        (token_r == "\\right\\lvert" or token_r == "\\right."):
        return True
    
    if (token_l == "\\left\\vert" or token_l == "\\left.") and \
        (token_r == "\\right\\vert" or token_r == "\\right."):
        return True
    
    if (token_l == "\\left\\Vert" or token_l == "\\left.") and \
        (token_r == "\\right\\Vert" or token_r == "\\right."):
        return True
    
    if (token_l == "\\left\\lfloor" or token_l == "\\left.") and \
        (token_r == "\\right\\rfloor" or token_r == "\\right."):
        return True
    
    if (token_l == "\\left\\lbrack" or token_l == "\\left.") and \
        (token_r == "\\right\\rbrack" or token_r == "\\right."):
        return True
    
    if (token_l == "\\left\\langle" or token_l == "\\left.") and \
        (token_r == "\\right\\rangle" or token_r == "\\right."):
        return True
    
    if (token_l == "\\left|" or token_l == "\\left.") and \
        (token_r == "\\right|" or token_r == "\\right."):
        return True
    
    if (token_l == "\\left\\|" or token_l == "\\left.") and \
        (token_r == "\\right\\|" or token_r == "\\right."):
        return True
    
    if (token_l == "\\left[" or token_l == "\\left.") and \
        (token_r == "\\right]" or token_r == "\\right."):
        return True
    
    if (token_l == "\\left]" or token_l == "\\left.") and \
        (token_r == "\\right[" or token_r == "\\right."):
        return True
    
    if (token_l == "\\left(" or token_l == "\\left.") and \
        (token_r == "\\right)" or token_r == "\\right."):
        return True
    
    if (token_l == "\\left)" or token_l == "\\left.") and \
        (token_r == "\\right(" or token_r == "\\right."):
        return True
    
    if (token_l == "\\left\\{" or token_l == "\\left.") and \
        (token_r == "\\right\\}" or token_r == "\\right."):
        return True
    
    if (token_l == "\\left\\}" or token_l == "\\left.") and \
        (token_r == "\\right\\{" or token_r == "\\right."):
        return True
    
    if (token_l == "\\left/" or token_l == "\\left.") and \
        (token_r == "\\right/" or token_r == "\\right."):
        return True
    
    return False


# fmt: on
def left_right_match(span_list):
    lr_stack = []
    lr_idx_stack = []
    for sub_span_idx, sub_span in enumerate(span_list):
        for token_idx, token in enumerate(sub_span):
            if token not in VALID_LEFT_TOKEN_LIST and token not in VALID_RIGHT_TOKEN_LIST:
                continue
            if len(lr_stack) == 0 and len(lr_idx_stack) == 0:
                lr_stack.append(token)
                lr_idx_stack.append((sub_span_idx, token_idx))
            else:
                if is_pair_left_right(lr_stack[-1], token):
                    lr_stack.pop()
                    lr_idx_stack.pop()
                else:
                    lr_stack.append(token)
                    lr_idx_stack.append((sub_span_idx, token_idx))
    return lr_stack, lr_idx_stack


def clean_span(node_list, node_tag_list):
    node_list_new = [node for node in node_list]

    span_all = [
        node_tag_list[idx]
        for idx in range(len(node_tag_list))
        if node_list[idx] != "&"
        and node_list[idx] != "\\\\"
        and "\\begin{array}" not in node_list[idx]
        and "\\end{array}" not in node_list[idx]
    ]

    span_list = list(set(span_all))

    valid_span_idx = [
        idx
        for idx in range(len(node_tag_list))
        if node_list[idx] != "&"
        and node_list[idx] != "\\\\"
        and "\\begin{array}" not in node_list[idx]
        and "\\end{array}" not in node_list[idx]
    ]

    for span in span_list:
        same_span_idx = [idx for idx in valid_span_idx if node_tag_list[idx] == span]

        span_list = [node_list[idx] for idx in same_span_idx]

        num_left = sum([count_left(span) for span in span_list])
        num_right = sum([count_right(span) for span in span_list])

        if num_left != num_right:
            span_list_fixed = []
            for span in span_list:
                span_splitted = split_with_left_right(span)
                span_splitted = [s for s in span_splitted if len(s.strip()) > 0]
                span_list_fixed.append(span_splitted)

            lr_stack, lr_idx_stack = left_right_match(span_list_fixed)
            for _lr, _lr_idx in zip(lr_stack, lr_idx_stack):
                if "\\left" in _lr:
                    span_list_fixed[_lr_idx[0]][_lr_idx[1]] = span_list_fixed[_lr_idx[0]][_lr_idx[1]] + " \\right."
                elif "\\right" in _lr:
                    span_list_fixed[_lr_idx[0]][_lr_idx[1]] = "\\left. " + span_list_fixed[_lr_idx[0]][_lr_idx[1]]

            span_list_fixed = ["".join(span_list) for span_list in span_list_fixed]
            for _idx, idx in enumerate(same_span_idx):
                node_list_new[idx] = span_list_fixed[_idx]

    return node_list_new


def fix_left_right_mismatch(latex: str):
    latex = latex.strip()
    node_list = split_with_delimiters(latex.strip())
    node_list = [node.strip() for node in node_list if len(node) > 0]
    array_list = tag_array(node_list)
    array_list = sorted(array_list, key=lambda x: len(x[-1]), reverse=True)
    node_tag_list = tag_element(node_list, array_list)
    node_list_new = clean_span(node_list, node_tag_list)
    fixed_latex = "".join(node_list_new)
    return fixed_latex


def try_match_equation_left_right(latex: str, debug: bool = False) -> str:
    if check_left_right(latex):
        return latex

    if not check_align(latex):
        return latex

    fixed_latex = fix_left_right_mismatch(latex)

    if debug:
        print(f"Trying to fix left-right mismatch in equation: {latex}")
        print(f"Fixed equation: {fixed_latex}")

    return fixed_latex
