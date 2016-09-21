FILE=$1
URL=https://people.eecs.berkeley.edu/~junyanz/projects/gvm/models/theano_dcgan/$FILE
OUTPUT_FILE=./models/$FILE

echo "Downloading the dcgan_theano model ($FILE)..."
wget -N $URL -O $OUTPUT_FILE
