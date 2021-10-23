import xml.etree.ElementTree as ET
import re, textwrap
tree = ET.parse('sms-20211023141012.xml')
root = tree.getroot()
with open ('metamorphosis_clean.txt','w') as f:
    for child in root:
        if 'type' in child.attrib and child.attrib['type']=='2':
            a_string = child.attrib['body']
            a_string = re.sub(r'http\S+', '', a_string)
            a_string = re.sub(r'/', '/ ', a_string)
            a_string = re.sub(r'\.', ' ', a_string)
            a_string = re.sub(r'\!', ' ', a_string)
            a_string = re.sub(r'\?', ' ', a_string)
            a_string = re.sub(r'  ', ' ', a_string)            
            a_string = re.sub(r'é', 'e', a_string)
            a_string = re.sub(r'ñ', 'n', a_string)
            alphanumeric = ""
            for character in a_string:
                if character.isalnum() or \
                character == ' ' or character == '\'' or \
                character == ':' or character == '-':
                    alphanumeric += character
            wrapper = textwrap.TextWrapper(width=65)
            alphanumeric = wrapper.fill(text=alphanumeric)
            alphanumeric = textwrap.dedent(text=alphanumeric)
            # print(alphanumeric)
            if len(alphanumeric) > 0:
                f.write(alphanumeric)
                f.write('\n')