## Importing Necessary Modules
import requests # to get image from the web
import shutil # to save it locally


output_dir = "sbu_images"
urls_file = 'SBU_captioned_photo_dataset_urls.txt'

with open(urls_file) as file:
    lines = [line.rstrip() for line in file]

print(lines[1])


## Set up the image URL and filename
image_url = "https://cdn.pixabay.com/photo/2020/02/06/09/39/summer-4823612_960_720.jpg"
filename = image_url.split("/")[-1]

# Open the url image, set stream to True, this will return the stream content.
r = requests.get(image_url, stream = True)

# Check if the image was retrieved successfully
if r.status_code == 200:
    # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
    r.raw.decode_content = True
    
    # Open a local file with wb ( write binary ) permission.
    with open(filename,'wb') as f:
        shutil.copyfileobj(r.raw, f)
        
    print('Image sucessfully Downloaded: ',filename)
else:
    print('Image Couldn\'t be retreived')



output_directory = 'sbu_images';
urls = textread('SBU_captioned_photo_dataset_urls.txt', '%s', -1);


if ~exist(output_directory, 'dir')
	mkdir(output_directory);
end

rand('twister', 123);
urls = urls(randperm(length(urls)));

%iterate for 30000
for i = 1 : 30000
	if ~exist(fullfile(output_directory, [regexprep(urls{i}(24, end), '/', '_')]))
		cmd = ['wget -t 3 -T 5 --quiet ' urls{i} ...
			   ' -O ' output_directory '/' regexprep(urls{i}(24, end), '/', '_')];
		unix(cmd);
        
		fprintf('%d. %s\n', i, urls{i});
	end	
end
'''