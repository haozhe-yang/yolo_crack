'''
通过解析xml文件，批量修改xml文件里的标签名称，比如把标签zero改成num
'''
import os.path
import glob
import xml.etree.ElementTree as ET

path = r'D:/yolo/crack/VOC2007/Annotations/'    #存储标签的路径，修改为自己的Annotations标签路径
for xml_file in glob.glob(path + '/*.xml'):
    ####### 返回解析树
	tree = ET.parse(xml_file)
	##########获取根节点
	root = tree.getroot()
	#######对所有目标进行解析
	for member in root.findall('object'):
		objectname = member.find('name').text
		if objectname == 'weironghe':      #原来的标签名字
			print(objectname)
			member.find('name').text = str('other')    #替换的标签名字
			tree.write(xml_file)
