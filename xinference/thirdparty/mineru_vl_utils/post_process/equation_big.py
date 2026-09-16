import re


def try_fix_equation_big(latex: str, debug: bool = False) -> str:

    # ------------------ \big{)} -> \big) ------------------ #
    original_latex = latex

    # \big
    latex = re.sub(r"\\big{\)}", r"\\big)", latex)
    latex = re.sub(r"\\big{\(}", r"\\big(", latex)
    latex = re.sub(r"\\big {\)}", r"\\big)", latex)
    latex = re.sub(r"\\big {\(}", r"\\big(", latex)

    # \bigr
    latex = re.sub(r"\\bigr{\)}", r"\\bigr)", latex)
    latex = re.sub(r"\\bigr{\(}", r"\\bigr(", latex)
    latex = re.sub(r"\\bigr {\)}", r"\\bigr)", latex)
    latex = re.sub(r"\\bigr {\(}", r"\\bigr(", latex)

    # \bigm
    latex = re.sub(r"\\bigm{\)}", r"\\bigm)", latex)
    latex = re.sub(r"\\bigm{\(}", r"\\bigm(", latex)
    latex = re.sub(r"\\bigm {\)}", r"\\bigm)", latex)
    latex = re.sub(r"\\bigm {\(}", r"\\bigm(", latex)

    # \bigl
    latex = re.sub(r"\\bigl{\)}", r"\\bigl)", latex)
    latex = re.sub(r"\\bigl{\(}", r"\\bigl(", latex)
    latex = re.sub(r"\\bigl {\)}", r"\\bigl)", latex)
    latex = re.sub(r"\\bigl {\(}", r"\\bigl(", latex)

    # \bigg
    latex = re.sub(r"\\bigg{\)}", r"\\bigg)", latex)
    latex = re.sub(r"\\bigg{\(}", r"\\bigg(", latex)
    latex = re.sub(r"\\bigg {\)}", r"\\bigg)", latex)
    latex = re.sub(r"\\bigg {\(}", r"\\bigg(", latex)

    # \biggr
    latex = re.sub(r"\\biggr{\)}", r"\\biggr)", latex)
    latex = re.sub(r"\\biggr{\(}", r"\\biggr(", latex)
    latex = re.sub(r"\\biggr {\)}", r"\\biggr)", latex)
    latex = re.sub(r"\\biggr {\(}", r"\\biggr(", latex)

    # \biggm
    latex = re.sub(r"\\biggm{\)}", r"\\biggm)", latex)
    latex = re.sub(r"\\biggm{\(}", r"\\biggm(", latex)
    latex = re.sub(r"\\biggm {\)}", r"\\biggm)", latex)
    latex = re.sub(r"\\biggm {\(}", r"\\biggm(", latex)

    # \biggl
    latex = re.sub(r"\\biggl{\)}", r"\\biggl)", latex)
    latex = re.sub(r"\\biggl{\(}", r"\\biggl(", latex)
    latex = re.sub(r"\\biggl {\)}", r"\\biggl)", latex)
    latex = re.sub(r"\\biggl {\(}", r"\\biggl(", latex)

    # \Big
    latex = re.sub(r"\\Big{\)}", r"\\Big)", latex)
    latex = re.sub(r"\\Big{\(}", r"\\Big(", latex)
    latex = re.sub(r"\\Big {\)}", r"\\Big)", latex)
    latex = re.sub(r"\\Big {\(}", r"\\Big(", latex)

    # \Bigr
    latex = re.sub(r"\\Bigr{\)}", r"\\Bigr)", latex)
    latex = re.sub(r"\\Bigr{\(}", r"\\Bigr(", latex)
    latex = re.sub(r"\\Bigr {\)}", r"\\Bigr)", latex)
    latex = re.sub(r"\\Bigr {\(}", r"\\Bigr(", latex)

    # \Bigm
    latex = re.sub(r"\\Bigm{\)}", r"\\Bigm)", latex)
    latex = re.sub(r"\\Bigm{\(}", r"\\Bigm(", latex)
    latex = re.sub(r"\\Bigm {\)}", r"\\Bigm)", latex)
    latex = re.sub(r"\\Bigm {\(}", r"\\Bigm(", latex)

    # \Bigl
    latex = re.sub(r"\\Bigl{\)}", r"\\Bigl)", latex)
    latex = re.sub(r"\\Bigl{\(}", r"\\Bigl(", latex)
    latex = re.sub(r"\\Bigl {\)}", r"\\Bigl)", latex)
    latex = re.sub(r"\\Bigl {\(}", r"\\Bigl(", latex)

    # \Bigg
    latex = re.sub(r"\\Bigg{\)}", r"\\Bigg)", latex)
    latex = re.sub(r"\\Bigg{\(}", r"\\Bigg(", latex)
    latex = re.sub(r"\\Bigg {\)}", r"\\Bigg)", latex)
    latex = re.sub(r"\\Bigg {\(}", r"\\Bigg(", latex)

    # \Biggr
    latex = re.sub(r"\\Biggr{\)}", r"\\Biggr)", latex)
    latex = re.sub(r"\\Biggr{\(}", r"\\Biggr(", latex)
    latex = re.sub(r"\\Biggr {\)}", r"\\Biggr)", latex)
    latex = re.sub(r"\\Biggr {\(}", r"\\Biggr(", latex)

    # \Biggm
    latex = re.sub(r"\\Biggm{\)}", r"\\Biggm)", latex)
    latex = re.sub(r"\\Biggm{\(}", r"\\Biggm(", latex)
    latex = re.sub(r"\\Biggm {\)}", r"\\Biggm)", latex)
    latex = re.sub(r"\\Biggm {\(}", r"\\Biggm(", latex)

    # \Biggl
    latex = re.sub(r"\\Biggl{\)}", r"\\Biggl)", latex)
    latex = re.sub(r"\\Biggl{\(}", r"\\Biggl(", latex)
    latex = re.sub(r"\\Biggl {\)}", r"\\Biggl)", latex)
    latex = re.sub(r"\\Biggl {\(}", r"\\Biggl(", latex)

    # ------------------ \big{\}} -> \big\} ------------------ #

    # \big
    latex = re.sub(r"\\big\{\\\}\}", r"\\big\\}", latex)
    latex = re.sub(r"\\big\{\\\{\}", r"\\big\\{", latex)
    latex = re.sub(r"\\big \{\\\}\}", r"\\big\\}", latex)
    latex = re.sub(r"\\big \{\\\{\}", r"\\big\\{", latex)

    # \bigr
    latex = re.sub(r"\\bigr\{\\\}\}", r"\\bigr\\}", latex)
    latex = re.sub(r"\\bigr\{\\\{\}", r"\\bigr\\{", latex)
    latex = re.sub(r"\\bigr \{\\\}\}", r"\\bigr\\}", latex)
    latex = re.sub(r"\\bigr \{\\\{\}", r"\\bigr\\{", latex)

    # \bigm
    latex = re.sub(r"\\bigm\{\\\}\}", r"\\bigm\\}", latex)
    latex = re.sub(r"\\bigm\{\\\{\}", r"\\bigm\\{", latex)
    latex = re.sub(r"\\bigm \{\\\}\}", r"\\bigm\\}", latex)
    latex = re.sub(r"\\bigm \{\\\{\}", r"\\bigm\\{", latex)

    # \bigl
    latex = re.sub(r"\\bigl\{\\\}\}", r"\\bigl\\}", latex)
    latex = re.sub(r"\\bigl\{\\\{\}", r"\\bigl\\{", latex)
    latex = re.sub(r"\\bigl \{\\\}\}", r"\\bigl\\}", latex)
    latex = re.sub(r"\\bigl \{\\\{\}", r"\\bigl\\{", latex)

    # \bigg
    latex = re.sub(r"\\bigg\{\\\}\}", r"\\bigg\\}", latex)
    latex = re.sub(r"\\bigg\{\\\{\}", r"\\bigg\\{", latex)
    latex = re.sub(r"\\bigg \{\\\}\}", r"\\bigg\\}", latex)
    latex = re.sub(r"\\bigg \{\\\{\}", r"\\bigg\\{", latex)

    # \biggr
    latex = re.sub(r"\\biggr\{\\\}\}", r"\\biggr\\}", latex)
    latex = re.sub(r"\\biggr\{\\\{\}", r"\\biggr\\{", latex)
    latex = re.sub(r"\\biggr \{\\\}\}", r"\\biggr\\}", latex)
    latex = re.sub(r"\\biggr \{\\\{\}", r"\\biggr\\{", latex)

    # \biggm
    latex = re.sub(r"\\biggm\{\\\}\}", r"\\biggm\\}", latex)
    latex = re.sub(r"\\biggm\{\\\{\}", r"\\biggm\\{", latex)
    latex = re.sub(r"\\biggm \{\\\}\}", r"\\biggm\\}", latex)
    latex = re.sub(r"\\biggm \{\\\{\}", r"\\biggm\\{", latex)

    # \biggl
    latex = re.sub(r"\\biggl\{\\\}\}", r"\\biggl\\}", latex)
    latex = re.sub(r"\\biggl\{\\\{\}", r"\\biggl\\{", latex)
    latex = re.sub(r"\\biggl \{\\\}\}", r"\\biggl\\}", latex)
    latex = re.sub(r"\\biggl \{\\\{\}", r"\\biggl\\{", latex)

    # \Big
    latex = re.sub(r"\\Big\{\\\}\}", r"\\Big\\}", latex)
    latex = re.sub(r"\\Big\{\\\{\}", r"\\Big\\{", latex)
    latex = re.sub(r"\\Big \{\\\}\}", r"\\Big\\}", latex)
    latex = re.sub(r"\\Big \{\\\{\}", r"\\Big\\{", latex)

    # \Bigr
    latex = re.sub(r"\\Bigr\{\\\}\}", r"\\Bigr\\}", latex)
    latex = re.sub(r"\\Bigr\{\\\{\}", r"\\Bigr\\{", latex)
    latex = re.sub(r"\\Bigr \{\\\}\}", r"\\Bigr\\}", latex)
    latex = re.sub(r"\\Bigr \{\\\{\}", r"\\Bigr\\{", latex)

    # \Bigm
    latex = re.sub(r"\\Bigm\{\\\}\}", r"\\Bigm\\}", latex)
    latex = re.sub(r"\\Bigm\{\\\{\}", r"\\Bigm\\{", latex)
    latex = re.sub(r"\\Bigm \{\\\}\}", r"\\Bigm\\}", latex)
    latex = re.sub(r"\\Bigm \{\\\{\}", r"\\Bigm\\{", latex)

    # \Bigl
    latex = re.sub(r"\\Bigl\{\\\}\}", r"\\Bigl\\}", latex)
    latex = re.sub(r"\\Bigl\{\\\{\}", r"\\Bigl\\{", latex)
    latex = re.sub(r"\\Bigl \{\\\}\}", r"\\Bigl\\}", latex)
    latex = re.sub(r"\\Bigl \{\\\{\}", r"\\Bigl\\{", latex)

    # \Bigg
    latex = re.sub(r"\\Bigg\{\\\}\}", r"\\Bigg\\}", latex)
    latex = re.sub(r"\\Bigg\{\\\{\}", r"\\Bigg\\{", latex)
    latex = re.sub(r"\\Bigg \{\\\}\}", r"\\Bigg\\}", latex)
    latex = re.sub(r"\\Bigg \{\\\{\}", r"\\Bigg\\{", latex)

    # \Biggr
    latex = re.sub(r"\\Biggr\{\\\}\}", r"\\Biggr\\}", latex)
    latex = re.sub(r"\\Biggr\{\\\{\}", r"\\Biggr\\{", latex)
    latex = re.sub(r"\\Biggr \{\\\}\}", r"\\Biggr\\}", latex)
    latex = re.sub(r"\\Biggr \{\\\{\}", r"\\Biggr\\{", latex)

    # \Biggl
    latex = re.sub(r"\\Biggl\{\\\}\}", r"\\Biggl\\}", latex)
    latex = re.sub(r"\\Biggl\{\\\{\}", r"\\Biggl\\{", latex)
    latex = re.sub(r"\\Biggl \{\\\}\}", r"\\Biggl\\}", latex)
    latex = re.sub(r"\\Biggl \{\\\{\}", r"\\Biggl\\{", latex)

    # ------------------ \big{|} -> \big\| ------------------ #

    # \big
    latex = re.sub(r"\\big\{\|\}", r"\\big|", latex)
    latex = re.sub(r"\\Big\{\|\}", r"\\Big|", latex)
    latex = re.sub(r"\\big \{\|\}", r"\\big|", latex)
    latex = re.sub(r"\\Big \{\|\}", r"\\Big|", latex)

    # \bigm
    latex = re.sub(r"\\bigm\{\|\}", r"\\bigm|", latex)
    latex = re.sub(r"\\Bigm\{\|\}", r"\\Bigm|", latex)
    latex = re.sub(r"\\bigm \{\|\}", r"\\bigm|", latex)
    latex = re.sub(r"\\Bigm \{\|\}", r"\\Bigm|", latex)

    # \bigr
    latex = re.sub(r"\\bigr\{\|\}", r"\\bigr|", latex)
    latex = re.sub(r"\\Bigr\{\|\}", r"\\Bigr|", latex)
    latex = re.sub(r"\\bigr \{\|\}", r"\\bigr|", latex)
    latex = re.sub(r"\\Bigr \{\|\}", r"\\Bigr|", latex)

    # \bigl
    latex = re.sub(r"\\bigl\{\|\}", r"\\bigl|", latex)
    latex = re.sub(r"\\Bigl\{\|\}", r"\\Bigl|", latex)
    latex = re.sub(r"\\bigl \{\|\}", r"\\bigl|", latex)
    latex = re.sub(r"\\Bigl \{\|\}", r"\\Bigl|", latex)

    # \bigg
    latex = re.sub(r"\\bigg\{\|\}", r"\\bigg|", latex)
    latex = re.sub(r"\\Bigg\{\|\}", r"\\Bigg|", latex)
    latex = re.sub(r"\\bigg \{\|\}", r"\\bigg|", latex)
    latex = re.sub(r"\\Bigg \{\|\}", r"\\Bigg|", latex)

    # \biggr
    latex = re.sub(r"\\biggr\{\|\}", r"\\biggr|", latex)
    latex = re.sub(r"\\Biggr\{\|\}", r"\\Biggr|", latex)
    latex = re.sub(r"\\biggr \{\|\}", r"\\biggr|", latex)
    latex = re.sub(r"\\Biggr \{\|\}", r"\\Biggr|", latex)

    # \biggm
    latex = re.sub(r"\\biggm\{\|\}", r"\\biggm|", latex)
    latex = re.sub(r"\\Biggm\{\|\}", r"\\Biggm|", latex)
    latex = re.sub(r"\\biggm \{\|\}", r"\\biggm|", latex)
    latex = re.sub(r"\\Biggm \{\|\}", r"\\Biggm|", latex)

    # \biggl
    latex = re.sub(r"\\biggl\{\|\}", r"\\biggl|", latex)
    latex = re.sub(r"\\Biggl\{\|\}", r"\\Biggl|", latex)
    latex = re.sub(r"\\biggl \{\|\}", r"\\biggl|", latex)
    latex = re.sub(r"\\Biggl \{\|\}", r"\\Biggl|", latex)

    # ------------------ \big{\|} -> \big\| ------------------ #

    # \big
    latex = re.sub(r"\\big\{\\\|\}", r"\\big\\|", latex)
    latex = re.sub(r"\\Big\{\\\|\}", r"\\Big\\|", latex)
    latex = re.sub(r"\\big \{\\\|\}", r"\\big\\|", latex)
    latex = re.sub(r"\\Big \{\\\|\}", r"\\Big\\|", latex)

    # \bigm
    latex = re.sub(r"\\bigm\{\\\|\}", r"\\bigm\\|", latex)
    latex = re.sub(r"\\Bigm\{\\\|\}", r"\\Bigm\\|", latex)
    latex = re.sub(r"\\bigm \{\\\|\}", r"\\bigm\\|", latex)
    latex = re.sub(r"\\Bigm \{\\\|\}", r"\\Bigm\\|", latex)

    # \bigr
    latex = re.sub(r"\\bigr\{\\\|\}", r"\\bigr\\|", latex)
    latex = re.sub(r"\\Bigr\{\\\|\}", r"\\Bigr\\|", latex)
    latex = re.sub(r"\\bigr \{\\\|\}", r"\\bigr\\|", latex)
    latex = re.sub(r"\\Bigr \{\\\|\}", r"\\Bigr\\|", latex)

    # \bigl
    latex = re.sub(r"\\bigl\{\\\|\}", r"\\bigl\\|", latex)
    latex = re.sub(r"\\Bigl\{\\\|\}", r"\\Bigl\\|", latex)
    latex = re.sub(r"\\bigl \{\\\|\}", r"\\bigl\\|", latex)
    latex = re.sub(r"\\Bigl \{\\\|\}", r"\\Bigl\\|", latex)

    # \bigg
    latex = re.sub(r"\\bigg\{\\\|\}", r"\\bigg\\|", latex)
    latex = re.sub(r"\\Bigg\{\\\|\}", r"\\Bigg\\|", latex)
    latex = re.sub(r"\\bigg \{\\\|\}", r"\\bigg\\|", latex)
    latex = re.sub(r"\\Bigg \{\\\|\}", r"\\Bigg\\|", latex)

    # \biggr
    latex = re.sub(r"\\biggr\{\\\|\}", r"\\biggr\\|", latex)
    latex = re.sub(r"\\Biggr\{\\\|\}", r"\\Biggr\\|", latex)
    latex = re.sub(r"\\biggr \{\\\|\}", r"\\biggr\\|", latex)
    latex = re.sub(r"\\Biggr \{\\\|\}", r"\\Biggr\\|", latex)

    # \biggm
    latex = re.sub(r"\\biggm\{\\\|\}", r"\\biggm\\|", latex)
    latex = re.sub(r"\\Biggm\{\\\|\}", r"\\Biggm\\|", latex)
    latex = re.sub(r"\\biggm \{\\\|\}", r"\\biggm\\|", latex)
    latex = re.sub(r"\\Biggm \{\\\|\}", r"\\Biggm\\|", latex)

    # \biggl
    latex = re.sub(r"\\biggl\{\\\|\}", r"\\biggl\\|", latex)
    latex = re.sub(r"\\Biggl\{\\\|\}", r"\\Biggl\\|", latex)
    latex = re.sub(r"\\biggl \{\\\|\}", r"\\biggl\\|", latex)
    latex = re.sub(r"\\Biggl \{\\\|\}", r"\\Biggl\\|", latex)

    # ------------------ \big{[} -> \big[ ------------------ #

    # \big
    latex = re.sub(r"\\big\{\]\}", r"\\big]", latex)
    latex = re.sub(r"\\Big\{\]\}", r"\\Big]", latex)
    latex = re.sub(r"\\big \{\]\}", r"\\big]", latex)
    latex = re.sub(r"\\Big \{\]\}", r"\\Big]", latex)

    latex = re.sub(r"\\big\{\[\}", r"\\big[", latex)
    latex = re.sub(r"\\Big\{\[\}", r"\\Big[", latex)
    latex = re.sub(r"\\big \{\[\}", r"\\big[", latex)
    latex = re.sub(r"\\Big \{\[\}", r"\\Big[", latex)

    # \bigm
    latex = re.sub(r"\\bigm\{\]\}", r"\\bigm]", latex)
    latex = re.sub(r"\\Bigm\{\]\}", r"\\Bigm]", latex)
    latex = re.sub(r"\\bigm \{\]\}", r"\\bigm]", latex)
    latex = re.sub(r"\\Bigm \{\]\}", r"\\Bigm]", latex)

    latex = re.sub(r"\\bigm\{\[\}", r"\\bigm[", latex)
    latex = re.sub(r"\\Bigm\{\[\}", r"\\Bigm[", latex)
    latex = re.sub(r"\\bigm \{\[\}", r"\\bigm[", latex)
    latex = re.sub(r"\\Bigm \{\[\}", r"\\Bigm[", latex)

    # \bigr
    latex = re.sub(r"\\bigr\{\]\}", r"\\bigr]", latex)
    latex = re.sub(r"\\Bigr\{\]\}", r"\\Bigr]", latex)
    latex = re.sub(r"\\bigr \{\]\}", r"\\bigr]", latex)
    latex = re.sub(r"\\Bigr \{\]\}", r"\\Bigr]", latex)

    latex = re.sub(r"\\bigr\{\[\}", r"\\bigr[", latex)
    latex = re.sub(r"\\Bigr\{\[\}", r"\\Bigr[", latex)
    latex = re.sub(r"\\bigr \{\[\}", r"\\bigr[", latex)
    latex = re.sub(r"\\Bigr \{\[\}", r"\\Bigr[", latex)

    # \bigl
    latex = re.sub(r"\\bigl\{\]\}", r"\\bigl]", latex)
    latex = re.sub(r"\\Bigl\{\]\}", r"\\Bigl]", latex)
    latex = re.sub(r"\\bigl \{\]\}", r"\\bigl]", latex)
    latex = re.sub(r"\\Bigl \{\]\}", r"\\Bigl]", latex)

    latex = re.sub(r"\\bigl\{\[\}", r"\\bigl[", latex)
    latex = re.sub(r"\\Bigl\{\[\}", r"\\Bigl[", latex)
    latex = re.sub(r"\\bigl \{\[\}", r"\\bigl[", latex)
    latex = re.sub(r"\\Bigl \{\[\}", r"\\Bigl[", latex)

    # \bigg
    latex = re.sub(r"\\bigg\{\]\}", r"\\bigg]", latex)
    latex = re.sub(r"\\Bigg\{\]\}", r"\\Bigg]", latex)
    latex = re.sub(r"\\bigg \{\]\}", r"\\bigg]", latex)
    latex = re.sub(r"\\Bigg \{\]\}", r"\\Bigg]", latex)

    latex = re.sub(r"\\bigg\{\[\}", r"\\bigg[", latex)
    latex = re.sub(r"\\Bigg\{\[\}", r"\\Bigg[", latex)
    latex = re.sub(r"\\bigg \{\[\}", r"\\bigg[", latex)
    latex = re.sub(r"\\Bigg \{\[\}", r"\\Bigg[", latex)

    # \biggr
    latex = re.sub(r"\\biggr\{\]\}", r"\\biggr]", latex)
    latex = re.sub(r"\\Biggr\{\]\}", r"\\Biggr]", latex)
    latex = re.sub(r"\\biggr \{\]\}", r"\\biggr]", latex)
    latex = re.sub(r"\\Biggr \{\]\}", r"\\Biggr]", latex)

    latex = re.sub(r"\\biggr\{\[\}", r"\\biggr[", latex)
    latex = re.sub(r"\\Biggr\{\[\}", r"\\Biggr[", latex)
    latex = re.sub(r"\\biggr \{\[\}", r"\\biggr[", latex)
    latex = re.sub(r"\\Biggr \{\[\}", r"\\Biggr[", latex)

    # \biggm
    latex = re.sub(r"\\biggm{\[}", r"\\biggm\[", latex)
    latex = re.sub(r"\\Biggm{\[}", r"\\Biggm\[", latex)
    latex = re.sub(r"\\biggm {\[}", r"\\biggm\[", latex)
    latex = re.sub(r"\\Biggm {\[}", r"\\Biggm\[", latex)

    latex = re.sub(r"\\biggm\{\]\}", r"\\biggm\]", latex)
    latex = re.sub(r"\\Biggm\{\]\}", r"\\Biggm\]", latex)
    latex = re.sub(r"\\biggm \{\]\}", r"\\biggm\]", latex)
    latex = re.sub(r"\\Biggm \{\]\}", r"\\Biggm\]", latex)

    # \biggl
    latex = re.sub(r"\\biggl\{\[\}", r"\\biggl\[", latex)
    latex = re.sub(r"\\Biggl\{\[\}", r"\\Biggl\[", latex)
    latex = re.sub(r"\\biggl \{\[\}", r"\\biggl\[", latex)
    latex = re.sub(r"\\Biggl \{\[\}", r"\\Biggl\[", latex)

    latex = re.sub(r"\\biggl\{\]\}", r"\\biggl\]", latex)
    latex = re.sub(r"\\Biggl\{\]\}", r"\\Biggl\]", latex)
    latex = re.sub(r"\\biggl \{\]\}", r"\\biggl\]", latex)
    latex = re.sub(r"\\Biggl \{\]\}", r"\\Biggl\]", latex)

    # ------------------ \big{\rangle} -> \big\rangle ------------------ #

    # \big
    latex = re.sub(r"\\big\{\\rangle\}", r"\\big\\rangle ", latex)
    latex = re.sub(r"\\big\{\\langle\}", r"\\big\\langle ", latex)
    latex = re.sub(r"\\big \{\\rangle\}", r"\\big\\rangle ", latex)
    latex = re.sub(r"\\big \{\\langle\}", r"\\big\\langle ", latex)

    # \bigr
    latex = re.sub(r"\\bigr\{\\rangle\}", r"\\bigr\\rangle ", latex)
    latex = re.sub(r"\\bigr\{\\langle\}", r"\\bigr\\langle ", latex)
    latex = re.sub(r"\\bigr \{\\rangle\}", r"\\bigr\\rangle ", latex)
    latex = re.sub(r"\\bigr \{\\langle\}", r"\\bigr\\langle ", latex)

    # \bigm
    latex = re.sub(r"\\bigm\{\\rangle\}", r"\\bigm\\rangle ", latex)
    latex = re.sub(r"\\bigm\{\\langle\}", r"\\bigm\\langle ", latex)
    latex = re.sub(r"\\bigm \{\\rangle\}", r"\\bigm\\rangle ", latex)
    latex = re.sub(r"\\bigm \{\\langle\}", r"\\bigm\\langle ", latex)

    # \bigl
    latex = re.sub(r"\\bigl\{\\rangle\}", r"\\bigl\\rangle ", latex)
    latex = re.sub(r"\\bigl\{\\langle\}", r"\\bigl\\langle ", latex)
    latex = re.sub(r"\\bigl \{\\rangle\}", r"\\bigl\\rangle ", latex)
    latex = re.sub(r"\\bigl \{\\langle\}", r"\\bigl\\langle ", latex)

    # \bigg
    latex = re.sub(r"\\bigg\{\\rangle\}", r"\\bigg\\rangle ", latex)
    latex = re.sub(r"\\bigg\{\\langle\}", r"\\bigg\\langle ", latex)
    latex = re.sub(r"\\bigg \{\\rangle\}", r"\\bigg\\rangle ", latex)
    latex = re.sub(r"\\bigg \{\\langle\}", r"\\bigg\\langle ", latex)

    # \biggr
    latex = re.sub(r"\\biggr\{\\rangle\}", r"\\biggr\\rangle ", latex)
    latex = re.sub(r"\\biggr\{\\langle\}", r"\\biggr\\langle ", latex)
    latex = re.sub(r"\\biggr \{\\rangle\}", r"\\biggr\\rangle ", latex)
    latex = re.sub(r"\\biggr \{\\langle\}", r"\\biggr\\langle ", latex)

    # \biggm
    latex = re.sub(r"\\biggm\{\\rangle\}", r"\\biggm\\rangle ", latex)
    latex = re.sub(r"\\biggm\{\\langle\}", r"\\biggm\\langle ", latex)
    latex = re.sub(r"\\biggm \{\\rangle\}", r"\\biggm\\rangle ", latex)
    latex = re.sub(r"\\biggm \{\\langle\}", r"\\biggm\\langle ", latex)

    # \biggl
    latex = re.sub(r"\\biggl\{\\rangle\}", r"\\biggl\\rangle ", latex)
    latex = re.sub(r"\\biggl\{\\langle\}", r"\\biggl\\langle ", latex)
    latex = re.sub(r"\\biggl \{\\rangle\}", r"\\biggl\\rangle ", latex)
    latex = re.sub(r"\\biggl \{\\langle\}", r"\\biggl\\langle ", latex)

    # \Big
    latex = re.sub(r"\\Big\{\\rangle\}", r"\\Big\\rangle ", latex)
    latex = re.sub(r"\\Big\{\\langle\}", r"\\Big\\langle ", latex)
    latex = re.sub(r"\\Big \{\\rangle\}", r"\\Big\\rangle ", latex)
    latex = re.sub(r"\\Big \{\\langle\}", r"\\Big\\langle ", latex)

    # \Bigr
    latex = re.sub(r"\\Bigr\{\\rangle\}", r"\\Bigr\\rangle ", latex)
    latex = re.sub(r"\\Bigr\{\\langle\}", r"\\Bigr\\langle ", latex)
    latex = re.sub(r"\\Bigr \{\\rangle\}", r"\\Bigr\\rangle ", latex)
    latex = re.sub(r"\\Bigr \{\\langle\}", r"\\Bigr\\langle ", latex)

    # \Bigm
    latex = re.sub(r"\\Bigm\{\\rangle\}", r"\\Bigm\\rangle ", latex)
    latex = re.sub(r"\\Bigm\{\\langle\}", r"\\Bigm\\langle ", latex)
    latex = re.sub(r"\\Bigm \{\\rangle\}", r"\\Bigm\\rangle ", latex)
    latex = re.sub(r"\\Bigm \{\\langle\}", r"\\Bigm\\langle ", latex)

    # \Bigl
    latex = re.sub(r"\\Bigl\{\\rangle\}", r"\\Bigl\\rangle ", latex)
    latex = re.sub(r"\\Bigl\{\\langle\}", r"\\Bigl\\langle ", latex)
    latex = re.sub(r"\\Bigl \{\\rangle\}", r"\\Bigl\\rangle ", latex)
    latex = re.sub(r"\\Bigl \{\\langle\}", r"\\Bigl\\langle ", latex)

    # \Bigg
    latex = re.sub(r"\\Bigg\{\\rangle\}", r"\\Bigg\\rangle ", latex)
    latex = re.sub(r"\\Bigg\{\\langle\}", r"\\Bigg\\langle ", latex)
    latex = re.sub(r"\\Bigg \{\\rangle\}", r"\\Bigg\\rangle ", latex)
    latex = re.sub(r"\\Bigg \{\\langle\}", r"\\Bigg\\langle ", latex)

    # \Biggr
    latex = re.sub(r"\\Biggr\{\\rangle\}", r"\\Biggr\\rangle ", latex)
    latex = re.sub(r"\\Biggr\{\\langle\}", r"\\Biggr\\langle ", latex)
    latex = re.sub(r"\\Biggr \{\\rangle\}", r"\\Biggr\\rangle ", latex)
    latex = re.sub(r"\\Biggr \{\\langle\}", r"\\Biggr\\langle ", latex)

    # \Biggl
    latex = re.sub(r"\\Biggl\{\\rangle\}", r"\\Biggl\\rangle ", latex)
    latex = re.sub(r"\\Biggl\{\\langle\}", r"\\Biggl\\langle ", latex)
    latex = re.sub(r"\\Biggl \{\\rangle\}", r"\\Biggl\\rangle ", latex)
    latex = re.sub(r"\\Biggl \{\\langle\}", r"\\Biggl\\langle ", latex)

    # \bigtimes -> \times
    latex = re.sub(r"\\bigtimes", r"\\times", latex)
    
    if debug and original_latex != latex:
        print(f"Fixed equation big from: {original_latex} to: {latex}")

    return latex
