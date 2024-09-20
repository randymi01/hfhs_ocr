import pytesseract

# download tesseract for windows here and add exe location here: https://github.com/UB-Mannheim/tesseract/wiki
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

from pdf2image import convert_from_path
import cv2
import numpy as np
import json
import os
import pickle
import click
import tqdm

text_fields = ["DATE_1", "MRN_1", "MRN_2", "NAME_1", "NAME_2","DATE_2", "MRN_2", "DOC_NAME", "DATE_2"]
decision_fields = ["CONTACT", "REPO", "DB"]

def preprocess_pdf(path_to_file):
    """
    PDF Preprocess. Will return two cv2 objects
    """
    pages = convert_from_path(path_to_file, dpi=300)

    if len(pages) != 2:
        raise Exception("Error: Expects PDF with two pages")
    
    p1_name = path_to_file[:-4] + '_pg1.jpg'
    p2_name = path_to_file[:-4] + '_pg2.jpg'

    pages[0].save(p1_name, 'JPEG')
    pages[1].save(p2_name, 'JPEG')

    return (p1_name, p2_name)

def preprocess_jpg(path_to_file, color = False):
    """
    JPG preprocessing with cv2
    applies greyscale, threshold
    """
    img = cv2.imread(path_to_file)

    os.remove(path_to_file)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        
    # Dilate corner image to enhance corner points
    corners = cv2.dilate(corners, None)

    # Threshold to get corner coordinates
    threshold = 0.01 * corners.max()
    corner_coords = np.where(corners > threshold)

    # Convert to list of (x, y) coordinates
    corner_points = list(zip(corner_coords[1], corner_coords[0]))

    rect = np.zeros((2, 2), dtype="int")
    s = np.sum(corner_points, axis=1)
    rect[0] = corner_points[np.argmin(s)]  # Top-left
    diff = np.diff(corner_points, axis=1)
    rect[1] = corner_points[np.argmin(diff)]  # Top-right

    # rect contains top corners

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables for the union bounding box
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    # Find bounding boxes and compute the union bounding box
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    if color:
        bounding_rect = img[int(rect[:,1].mean()) : max_y, rect[0][0] : rect[1][0]]
    else:
        bounding_rect = thresh[int(rect[:,1].mean()) : max_y, rect[0][0] : rect[1][0]]

    return cv2.resize(bounding_rect, (2550, 3300))


def preprocess(path_to_file, color = False):
    """"
    Takes in pdf file and returns a cv2 array applying preprocessing steps above
    If pdf, expects that it is a file containing pg 10, 11 sequentially
    """
    if path_to_file.endswith('.pdf'):
        p1_name, p2_name = preprocess_pdf(path_to_file)
        return [preprocess_jpg(p1_name, color = color),
                preprocess_jpg(p2_name, color = color)]

    #elif path_to_file.endswith('.jpg'):
    #    return preprocess_jpg(path_to_file, color = color)
    
    
    raise Exception("Error: Invalid Input. Must be of type (*.pdf)")

def all_numeric(i_str : str) -> bool:
    for i in i_str:
        if not i.isdigit():
            return False
    return True

def get_text(preprocessed_img, annotations, validation_mode = False, conf_min = 50):
    text = {}
    for shape in annotations['shapes']:
        bbox_dim = shape['points']
        bbox = preprocessed_img[int(bbox_dim[0][1]) : int(bbox_dim[1][1]),
                int(bbox_dim[0][0]) : int(bbox_dim[1][0])]
        
        #plt.figure(figsize=(12, 8))
        #plt.imshow(bbox)
        #plt.title(shape['label'])
        #plt.show()
        
        if shape['label'] in text_fields and not validation_mode:
            output = pytesseract.image_to_string(bbox, lang = 'eng')
            data = pytesseract.image_to_data(bbox, lang='eng', output_type=pytesseract.Output.DATAFRAME)
            # Filter out rows with no text
            data = data[data['text'].notna()]
            if len(data) == 0:
                text[shape['label']] = None
            else:
                dval = data.sort_values('conf', ascending = False).iloc[0]
                output_val = output.strip()
                if dval.conf < conf_min:  
                    output_val = None
                else:
                    # MRN Validation
                    if shape['label'].startswith('MRN'):
                        if not all_numeric(output_val):
                            output_val = None        
                            
                text[shape['label']] = output_val      
        else:
            #report average weight
            text[shape['label']] = bbox.flatten().mean()

    return text

def collect_texts(pg1, pg2, pg1_annotations, 
                  pg2_annotations, base_weight_pg1, base_weight_pg2):
    fields = {}
    # process pg1
    for pg, annotation, base_weight in zip([pg1, pg2], [pg1_annotations, pg2_annotations], [base_weight_pg1, base_weight_pg2]):
        pg_text = get_text(pg, annotation)
        for field, val in pg_text.items():
            
            if field in text_fields:
                fields[field] = pg_text[field]
                continue
            root = field.split('_')[0]
            if root in decision_fields:
                if (pg_text[root + '_R'] / base_weight[root + '_R']) > (pg_text[root + '_AG'] / base_weight[root + '_AG']):
                    fields[root] = 'REFUSED'
                else:
                    fields[root] = 'AGREED'
                continue

            # all that is left is exit fields
            if (pg_text[field] / base_weight[field]) > 1.1:
                fields[field] = "EXISTS"
            else:
                fields[field] = "MISSING"
        
    return fields

@click.command()
@click.option('-i','--input_path', required=True, help='path to single pdf of last two pages of patient consent form or directory containing pdfs')
@click.option('-o','--output_path', required=True, help='relative folder path for where json data should be stored')
@click.option('-d', '--delete', required=False, default=False, help='delete pdfs', show_default = True)
def main(input_path, output_path, delete):
    with open("templates/pg1_annotations_n.json", 'r') as file:
        pg1_annotations = json.load(file)
    with open("templates/pg2_annotations_n.json", 'r') as file:
        pg2_annotations = json.load(file)
    with open("templates/base_weight_pg1.pkl", "rb") as file:
        base_weight_pg1 = pickle.load(file)
    with open("templates/base_weight_pg2.pkl", "rb") as file:
        base_weight_pg2 = pickle.load(file)
    os.makedirs(output_path, exist_ok=True)
    if os.path.isdir(input_path):
        for filename in tqdm(os.listdir(input_path)):
            file_path = os.path.join(input_path, filename)
            pg1, pg2 = preprocess(file_path)
            fields = collect_texts(pg1, pg2, pg1_annotations, pg2_annotations, base_weight_pg1, base_weight_pg2)

            output_file_path = os.path.join(output_path, f"{os.path.splitext(filename)[0]}_fields.json")
            with open(output_file_path, 'w') as output_file:
                json.dump(fields, output_file, indent=4)
            if delete:
                os.remove(file_path)
    else:
        pg1, pg2 = preprocess(input_path)
        fields = collect_texts(pg1, pg2, pg1_annotations, pg2_annotations, base_weight_pg1, base_weight_pg2)

        output_file_path = os.path.join(output_path, f"{os.path.splitext(input_path)[0]}_fields.json")
        with open(output_file_path, 'w') as output_file:
            json.dump(fields, output_file, indent=4)
        if delete:
            os.remove(file_path)   

if __name__ == '__main__':
    main()
    

    