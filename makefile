
zip:
	zip 23-xvlach22.zip docs/dokumentace.pdf install.sh requirements.txt src/*.py \
	dataset/ssc_29-06-16/set_1000*/* models/torch_gru_8hid_250batch_21000epochs.pt \
	python-scripts/Dataset_creating.ipynb

clean:
	rm 23-xvlach22.zip
