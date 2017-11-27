NAME=Cover_Page

$(NAME).pdf: $(NAME).tex
	pdflatex "$<"

clean:
	rm -f $(NAME).dvi *.pdf *.lof *.log *.lot *.aux *.blg *.bbl *.toc
