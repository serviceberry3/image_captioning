import requests #lets us get img from web
import shutil #lets us easily save imgs locally


#specify dir in which we want to save imgs, and the txt file containing all of the img urls
output_dir = "./val/images/sbu/"
urls_file = 'SBU_captioned_photo_dataset_urls.txt'


#open the txt file and read all the url strings into a list
with open(urls_file) as file:
    lines = [line.rstrip() for line in file]


#print(lines[1][-14:-4])

num_downloaded = 0

for url in lines[10000:]:
    # Open the url image, set stream to True, this will return the stream content.
    r = requests.get(url, stream = True)

    # Check if the image was retrieved successfully
    if r.status_code == 200:
        #set local filename to be <10-digit img id>.jpg
        filename = url[-14:]

        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True
        
        # Open a local file with wb ( write binary ) permission.
        with open(output_dir + filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
            
        print('Image sucessfully downloaded and saved as {}'.format(output_dir + filename))

        num_downloaded += 1
    else:
        print('Image could not be retrieved')


print("successfully downloaded {} total imgs".format(num_downloaded))