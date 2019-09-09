.PHONY: dirs data requirements

dirs:
	bash -c "mkdir -p data/{raw,split,processed,models}"

data:
	curl \
	https://data.consumerfinance.gov/api/views/s6ew-h6mp/rows.csv?accessType=DOWNLOAD \
	> data/raw/consumer_complaints.csv

requirements:
	pip3 install -r requirements.txt
