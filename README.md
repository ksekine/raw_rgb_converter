# Raw / RGB converter  

## Set environment  

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Raw (RGGB) to RGB  

```
python raw2rgb.py --load_dir [LOAD_DIR] --save_dir [SAVE_DIR] --width [IMAGE_WIDTH] --height [IMAGE_HEIGHT] --ext [SEARCH_EXTENSION]
```

## RGB to Raw (RGGB)  

**Sampling pixels are not enough.**  

```
python rgb2raw.py --load_dir [LOAD_DIR] --save_dir [SAVE_DIR] --ext [SEARCH_EXTENSION]
```
