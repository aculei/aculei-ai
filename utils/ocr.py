from PIL import Image
import pytesseract
import re
from datetime import datetime

def extract_temperature(img):
    width, height = img.size
    
    # keep only the bottom 1/3 of the image
    left = 0
    right = width
    lower = int(height * 17 / 18)  

    cropped_image = img.crop((left, lower, right, height))

    # psm 11: sparse text. Find as much text as possible in no particular order.
    args = ["--psm 11"]    
    result = pytesseract.image_to_string(image=cropped_image, lang='eng', config=" ".join(args))

    regex = r"([0-9]+ ?°c)"
    pattern = re.compile(regex, re.IGNORECASE)
    matches = re.findall(pattern, result)

    temperature = matches[0] if matches else None
    
    if temperature:
        temperature = re.sub(r'\D', '', temperature)
    
    return temperature

def extract_date(img):
    width, height = img.size
    
    # keep only the bottom 1/3 of the image
    left = 0
    right = width
    lower = int(height * 17 / 18)  

    cropped_image = img.crop((left, lower, right, height))

    # psm 11: sparse text. Find as much text as possible in no particular order.
    args = ["--psm 11"]    
    text = pytesseract.image_to_string(image=cropped_image, lang='eng', config=" ".join(args))

    date_pattern = r'\b\d{4}[-/]\d{2}[-/]\d{2}\b|\b\d{2}[-/]\d{2}[-/]\d{4}\b'    
    time_pattern = r'\b\d{1,2}:\d{2}:\d{2}\b'
    date_matches = re.findall(date_pattern, text)
    time_matches = re.findall(time_pattern, text)

    formatted_date = None
    time = None
    if date_matches:
        parsed_date = parse_date(date_matches[0])
        if parsed_date is not None:
            formatted_date = parsed_date.strftime("%Y-%m-%d")
        time = time_matches[0] if time_matches else None
    
    dt = datetime.strptime(formatted_date + " " + time, "%Y-%m-%d %H:%M:%S") if formatted_date and time else None
    return str(dt) if dt else None

def extract_camera(img):
    width, height = img.size
    
    # keep only the bottom 1/3 of the image
    left = 0
    right = width
    lower = int(height * 17 / 18)  

    cropped_image = img.crop((left, lower, right, height))

    # psm 11: sparse text. Find as much text as possible in no particular order.
    args = ["--psm 11"]    
    text = pytesseract.image_to_string(image=cropped_image, lang='eng', config=" ".join(args))

    # camera pattern must match CAM followed by 1 digit ignoring case
    camera_pattern = r'\bCAM\d\b'
    camera_matches = re.findall(camera_pattern, text)

    camera = str.upper(camera_matches[0]) if camera_matches else None
    return camera

def parse_date(date_string):
    for format in ["%Y/%m/%d", "%d/%m/%Y"]:
        try:
            return datetime.strptime(date_string, format)
        except ValueError:
            pass
    return None