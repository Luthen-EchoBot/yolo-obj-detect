# Instruction

## Setup
```bash
git clone https://github.com/Luthen-EchoBot/yolo-obj-detect.git
cd yolo-obj-detect
pyenv install 3.10.14
pyenv shell 3.10.14
python -m venv venv
source venv/bin/activate
pip install -U ultralytics
# pip install -r requirements.txt
```

## Use
```bash
source venv/bin/activate # skip if terminal starts with "(venv)"
python categorize_terminal.py
# use 'q' to close program
```

# Yolo object detection
## Setup
```bash
git clone https://github.com/Luthen-EchoBot/yolo-obj-detect.git
cd yolo-obj-detect
pyenv virtualenv 3.11.9 venv311
pyenv shell 3.10.14
python -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install torch torchvision --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v5.1
pip install ultralytics
```

## Use
# Connecter a la jetson
```bash
ssh pi@10.105.1.167
```
Password: geicar
```bash
ssh jetson@192.168.1.10
```
Password: jetson
```bash
cd AI/HumanDetecion/yolo-obj-detect
source venv/bin/activate
python track_graphic.py
```
