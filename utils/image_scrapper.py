from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import requests
import pandas as pd
import os
import time

user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

options = Options()
# options.add_experimental_option("detach", True)
options.headless = True
options.add_argument(f'user-agent={user_agent}')
options.add_argument("--window-size=1920,1080")
options.add_argument('--ignore-certificate-errors')
options.add_argument('--allow-running-insecure-content')
options.add_argument("--disable-extensions")
options.add_argument("--proxy-server='direct://'")
options.add_argument("--proxy-bypass-list=*")
options.add_argument("--start-maximized")
options.add_argument('--disable-gpu')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--no-sandbox')
options.add_argument("--headless")

driver = webdriver.Chrome(options=options)

def getImages(data, file_name):

    img_to_save = '/Users/viru/Documents/GitHub/Micro-Influencer-Recommendations/brands-dataset/automobile/' + file_name.split('.')[0]

    # Create the folder if it does not exist
    if not os.path.exists(img_to_save):
        os.makedirs(img_to_save)
        print(f"Folder '{img_to_save}' created successfully")

    # Iterate over each row in the grouped DataFrame
    for index, row in data.iterrows():
        owner_username = row['ownerUsername']
        urls = row['url']

        # infl_folder = img_to_save + '/' + owner_username

        # # Create the folder if it does not exist
        # if not os.path.exists(infl_folder):
        #     os.makedirs(infl_folder)
        #     print(f"Folder '{infl_folder}' created successfully")

        for url in urls:
            # Navigate to the webpage containing the image
            driver.get(url)

            time.sleep(3)
            try:
                # Locate the HTML element that corresponds to the image you want to download
                div_element = driver.find_element(By.CLASS_NAME, '_aagv')
                image_element = div_element.find_element(By.TAG_NAME, 'img')

                image_url = image_element.get_attribute("src")

                # Use requests to download the image
                response = requests.get(image_url)
                if response.status_code == 200:
                    # Construct the full file path
                    filename = os.path.join(img_to_save, f"image_{urls.index(url)}.jpg")  # Use a unique filename
                    with open(filename, "wb") as f:
                        f.write(response.content)
                    print(f"Image downloaded successfully and saved as {filename}")
                else:
                    print("Failed to download image")
            except Exception as e:
                print(f"Skipping image: {urls.index(url)}")

    # Close the WebDriver
    driver.quit()


# automobile_dir = '/Users/viru/Documents/GitHub/Micro-Influencer-Recommendations/influencer-dataset/categories/automobile'
# automobile_influencers = ['busfansadoor', 'lifeathgs', 'frank_highway', 'gulfoil.india', 'thechoprafoundation', 'eeco_carlover', 'gsrtc_unplugged', '_etauto', 'autos_vlog', 'ksrtc.unofficial', '8feet6wheels', 'mitchgilbertracing', 'therajmahalpalace', 'auditradition', 'divyagursahani', 'globalspaindia', 'zohacastelino', 'lucas74cruz', 's.peterhansel', 'raashotels', 'benoittreluyer', 'whatesh', '_nehajha_', 'humsufi_insta', 'apacheindianhq', 'anushriyagulati', 'nataliemicah', 'broken.asphalt', 'mr_harley_kid', 'motorcycletales', 'indiabikeweek', 'indiabikeweek', 'theweaero', 'ritambhatnagar', 'prateek.mahesh', 'labelnityabajaj', 'vaibhavrajgupta', 'theenigmaticdrummer', 'shankuraj_konwarmusical', 'sunita_rajwar', 'pardeep_kumar11', 'aline_krauter', 'mizoram_tourism', 'awaaraphotographer', 'aemotorshow', 'mad_over_travell', 'titusupputuru', 'hormazdsorabjee', 'gawdesslike', 'autocar_official', 'jitendravaswani', 'siddique_sculpture', 'martyco.in', '_megharoy_', 'autoxmag', '_khantaha_', 'motorcyclenews', 'triumphgermany', 'mr_bajaj', 'suparbiker', 'max.hwrd', 'broken.asphalt', 'wrapcraft', 'kevinthomasck', 'indiabikeweek', 'winter_wanderer', 'gods_own_jeepers', 'nos_motografia', 'framefusion_', 'indiawithinsia', 'jeep_at', 'jeepuaete', 'punisher7547', 'bhatius', 'kartick.wildlifesos', 'jeepnz', 'rohitmane93', 'iamunimo', 'theterratribe', 'modi_equestrian', 'sorabhpant', 'exploringwithroy', 'tatapowercompanyltd', 'sudhanshuchandra', 'kreate_with_kashif', 'futureskillsprime', 'jeswinrebello', 'akclixx', 'prithwi', 'sainath_rockstar', 'oaktreesport', 'cyrusdhabhar', 'arppithaandaa', 'gayathryrajiv', 'rohitmane93', 'madhura__balaji', 'rachit.hirani', 'turbochargedmag', 'almostaayush', 'mathieucesar', 'purvapandit', 'nitishwaila', 'motorworldindia', 'newindianexpress', 'potter.sophie', 'carindia_mag', 'lucavanassche', 'lucavanassche', 'laurensvandenacker', 'deepika_sethi_1', 'sajad_machu', 'sidpatankar', 'motorplanetofficial_com', 'carswithkaran', 'volkswagen_quadrados', 'bandishprojekt', 'hormazdsorabjee', 'gti_lovers', 'oaffmusic', 'kunalrawaldstress', 'thegtproduction']

automobile_brand_dir = '/Users/viru/Documents/GitHub/Micro-Influencer-Recommendations/brands-dataset/automobile'

xlsx_files = [file for file in os.listdir(automobile_brand_dir) if file.endswith('.xlsx')]

for file_name in xlsx_files:
    file_path = os.path.join(automobile_brand_dir, file_name)

    df = pd.read_excel(file_path)

    df = df[['ownerUsername', 'url']]
    grouped_df = df.groupby('ownerUsername')['url'].agg(list).reset_index()

    getImages(grouped_df, file_name)
