import os
import json
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString


if not os.path.exists('VIDOR80/Annotations/train'):
    os.makedirs('VIDOR80/Annotations/train')
if not os.path.exists('VIDOR80/Annotations/val'):
    os.makedirs('VIDOR80/Annotations/val')


def make_xml(preids, im_id, ih, iw, boxes, tids, clses, generated):
    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = str(preids)
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = str(im_id)
    node_source = SubElement(node_root, 'source')
    node_dataset = SubElement(node_source, 'dataset')
    node_dataset.text = 'VIDOR80'

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(iw)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(ih)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    for i, box in enumerate(boxes):
        node_object = SubElement(node_root, 'object')
        node_trackid = SubElement(node_object, 'trackid')
        node_trackid.text = str(tids[i])
        node_name = SubElement(node_object, 'name')
        node_name.text = clses[i]

        xmin, ymin, xmax, ymax = box
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(xmin)
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(ymin)
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(xmax)
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(ymax)

        node_occluded = SubElement(node_object, 'occluded')
        node_occluded.text = '0'
        node_generated = SubElement(node_object, 'generated')
        node_generated.text = str(generated[i])

    xml = tostring(node_root)
    dom = parseString(xml)
    return dom


def video2anno(mode):
    json_num = 0
    if mode == 'train':
        root_dir = 'annotation/training/'
    else:
        root_dir = 'annotation/validation/'
    for preids in os.listdir(root_dir):
        pre_path = root_dir + preids
        pre_xmls_path = 'VIDOR80/Annotations/'+mode+'/' + preids
        if not os.path.exists(pre_xmls_path):
            os.makedirs(pre_xmls_path)
        for json_name in os.listdir(pre_path):
            print(json_num, json_name)
            json_id = json_name[:-5]
            xml_dir = os.path.join(pre_xmls_path, json_id)
            if not os.path.exists(xml_dir):
                os.makedirs(xml_dir)
            json_num += 1
            if json_num % 100 == 0:
                print(json_num)

            json_path = os.path.join(pre_path, json_name)
            with open(json_path, 'r') as f:
                data = json.load(f)

            ih = data['height']
            iw = data['width']
            objs = data['subject/objects']

            for num, frame in enumerate(data['trajectories']):
                #if not num % 10 == 0:
                #    continue
                clses = []
                boxes = []
                tids = []
                generated = []
                for box in frame:
                    #print(box)
                    for obj in objs:
                        if box['tid'] == obj['tid']:
                            clses.append(obj['category'])
                    xmin = box['bbox']['xmin']
                    ymin = box['bbox']['ymin']
                    xmax = box['bbox']['xmax']
                    ymax = box['bbox']['ymax']
                    boxes.append([xmin, ymin, xmax, ymax])
                    tid = box['tid'] 
                    tids.append(tid)
                    generated.append(frame[0]['generated'])
                dom = make_xml(preids, num, ih, iw, boxes, tids, clses, generated)
                xml_name = os.path.join(xml_dir, str(num)+'.xml')
                with open(xml_name, 'wb') as f:
                    f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))


if __name__ == '__main__':
    video2anno('train')
    video2anno('val')
