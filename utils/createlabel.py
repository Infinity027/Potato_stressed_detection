import os 
import xml.etree.ElementTree as ET
import argparse

parser = argparse.ArgumentParser(description="Provide following parameters")

parser.add_argument("-i", '--input', dest="input_path", required=True, type=str,
                    help="input directory")

args = parser.parse_args()

if not os.path.exists(args.input_path):
    raise FileNotFoundError(f"The input path '{args.input_path}' does not exist.")

class_name_to_id_mapping = {"healthy":0, "stressed":1}
i = 0

for xml_name in os.listdir(args.input_path):
    if not (xml_name.endswith('.xml')):
        continue
    i += 1
    tree = ET.parse(os.path.join(args.input_path, xml_name))
    root = tree.getroot()
    filename = root.find('filename').text
    size = root.find('size')
    image_w = int(size.find('width').text)
    image_h = int(size.find('height').text)
    print_buffer = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        try:
            class_id = class_name_to_id_mapping[label]
        except KeyError:
            print("Invalid Class",label)
            print(xml_name)
        
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (xmin + xmax) / 2 
        b_center_y = (ymin + ymax) / 2
        b_width    = (xmax - xmin)
        b_height   = (ymax - ymin)
        # Normalise the co-ordinates by the dimensions of the image
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h 
        #Write the bbox details to the file 
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
        
    # Name of the file which we have to save 
    save_file_name = os.path.join(args.input_path.replace("rgb","labels"),xml_name.replace("xml","txt"))
    print("\n".join(print_buffer), file= open(save_file_name, "w"))
print(save_file_name)
print(i)