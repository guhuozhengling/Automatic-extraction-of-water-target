git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

cd ~
mkdir .kaggle

cd /content/Automatic-extraction-of-water-target/
cp kaggle.json ~/.kaggle
kaggle datasets download -d ltxltx/shuiti
unzip shuiti.zip

sh train.sh
