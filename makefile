
zip:
	rm -rf src/__pycache__
	zip 23-xvlach22.zip dokumentace.pdf install.sh requirements.txt \
		dataset/ssc_29-06-16/set_*/* dataset/long_words_test.txt \
		docs/*.png docs/main.tex \
		models/*.pt \
		src/*.py python-scripts/Dataset_creating.ipynb

docu:
	pdflatex docs/main.tex

afterdocu:
	rm -f main.aux
	rm -f main.out
	rm -f main.log
	mv main.pdf dokumentace.pdf


clean:
	# rm 23-xvlach22.zip
