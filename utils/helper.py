import os
import pandas as pd

automobile_dir = '/Users/viru/Documents/GitHub/Micro-Influencer-Recommendations/influencer-dataset/categories/automobile'
automobile_influencers = ['busfansadoor', 'lifeathgs', 'frank_highway', 'gulfoil.india', 'thechoprafoundation', 'eeco_carlover', 'gsrtc_unplugged', '_etauto', 'autos_vlog', 'ksrtc.unofficial', '8feet6wheels', 'mitchgilbertracing', 'therajmahalpalace', 'auditradition', 'divyagursahani', 'globalspaindia', 'zohacastelino', 'lucas74cruz', 's.peterhansel', 'raashotels', 'benoittreluyer', 'whatesh', '_nehajha_', 'humsufi_insta', 'apacheindianhq', 'anushriyagulati', 'nataliemicah', 'broken.asphalt', 'mr_harley_kid', 'motorcycletales', 'indiabikeweek', 'indiabikeweek', 'theweaero', 'ritambhatnagar', 'prateek.mahesh', 'labelnityabajaj', 'vaibhavrajgupta', 'theenigmaticdrummer', 'shankuraj_konwarmusical', 'sunita_rajwar', 'pardeep_kumar11', 'aline_krauter', 'mizoram_tourism', 'awaaraphotographer', 'aemotorshow', 'mad_over_travell', 'titusupputuru', 'hormazdsorabjee', 'gawdesslike', 'autocar_official', 'jitendravaswani', 'siddique_sculpture', 'martyco.in', '_megharoy_', 'autoxmag', '_khantaha_', 'motorcyclenews', 'triumphgermany', 'mr_bajaj', 'suparbiker', 'max.hwrd', 'broken.asphalt', 'wrapcraft', 'kevinthomasck', 'indiabikeweek', 'winter_wanderer', 'gods_own_jeepers', 'nos_motografia', 'framefusion_', 'indiawithinsia', 'jeep_at', 'jeepuaete', 'punisher7547', 'bhatius', 'kartick.wildlifesos', 'jeepnz', 'rohitmane93', 'iamunimo', 'theterratribe', 'modi_equestrian', 'sorabhpant', 'exploringwithroy', 'tatapowercompanyltd', 'sudhanshuchandra', 'kreate_with_kashif', 'futureskillsprime', 'jeswinrebello', 'akclixx', 'prithwi', 'sainath_rockstar', 'oaktreesport', 'cyrusdhabhar', 'arppithaandaa', 'gayathryrajiv', 'rohitmane93', 'madhura__balaji', 'rachit.hirani', 'turbochargedmag', 'almostaayush', 'mathieucesar', 'purvapandit', 'nitishwaila', 'motorworldindia', 'newindianexpress', 'potter.sophie', 'carindia_mag', 'lucavanassche', 'lucavanassche', 'laurensvandenacker', 'deepika_sethi_1', 'sajad_machu', 'sidpatankar', 'motorplanetofficial_com', 'carswithkaran', 'volkswagen_quadrados', 'bandishprojekt', 'hormazdsorabjee', 'gti_lovers', 'oaffmusic', 'kunalrawaldstress', 'thegtproduction']

electronics_dir = '/Users/viru/Documents/GitHub/Micro-Influencer-Recommendations/influencer-dataset/categories/electronics'
electronics_influencers = [
    'ibcshow',
    'bhaatu',
    'ggfevents',
    'reddotdesignaward',
    'mr_matt_lee',
    'metrodoodle',
    'proart',
    'escapestudios',
    'anandeshwardwivedi',
    'nupurnagpal02',
    'noctua_at',
    'rubenderonde',
    'withrepost',
    'ankitpanth',
    'wander_leen',
    'gamertweak',
    'ozgewhocodes',
    'havoknationin',
    'webdeveloper.io',
    'ankurtewari',
    'ahmedaftabnaqvi',
    'gamerconnect.in',
    'lorrainenam',
    'nayantaraparikh',
    'vineetpanchhi',
    'foxyoxie',
    'atelierhabib',
    'sudhirbbxr',
    'innanieariffin',
    'christianbong',
    'tianchad',
    'emmakateco',
    'sarafvibh',
    'tech4gaming',
    'western.pa.trains',
    'smixity',
    'c9emz',
    'cupahnoodle',
    'aimstv_',
    'btmclive',
    'emuhleet',
    'pandaglobal.pg',
    'uminokaiju',
    'blaustoise',
    'palladiumahmedabad',
    'mygcarekerala',
    'heysuhith',
    'little.anay',
    'raveena.vishal',
    'medhavista',
    'skils_shruti',
    'nogueraalberto',
    'nanni_sekhon07',
    'touchwoodautomations',
    'rovers_trail',
    'amd.india',
    'tech_vani',
    'theesportsclub',
    'rakshit.tandon',
    'islandconservation',
    'mc_heam',
    'team_centa',
    'technoserve',
    'toosid',
    'mvdhav',
    'drprernakohli.in',
    'rohanalbal',
    'balmont.art',
    'veerendrajillella',
    'sanjamarusic',
    'chalta_firta_photographer',
    'amateurphotoclicker',
    'thegullygrapher',
    'the.shailendrasingh',
    'pixelapse',
    'chrmsmusic',
    'vollut',
    '_mr.lensman',
    'aarohantiwari',
    'rishabhpaliwal26',
    'photowale.bapu',
    'rjphotography2149',
    'nayanmarathe',
    'gauravvfilms',
    'the_flattering_focus',
    'vrutik_vr_',
    'miteshdop',
    'firangi_photowala',
    'musafir_dil_harshi',
    'neetashankar',
    'mahendrabakle',
    'egonslacis',
    'nitinaroraphotography',
    'itsnitinarora',
    'giovannigenzini',
    'awaara_in_the_city',
    'romaganeshphotography',
    'jioworldplaza',
    'theupsidespace',
    'labelpsb',
    'vijaydayal',
    'davbowman',
    'stage223',
    'manjotroyal',
    'andersonandwilson',
    'mohitkapoormk',
    'twogetherstudios.in',
    'stevekouta',
    'saishravanam',
    'ronansiri',
    'retrogamesbar',
    'hispanic_federation',
    'crunchyrollexpo',
    'ginzasonypark',
    'wannadancemovie',
    'granturismomovie',
    'clmazin',
    'sonyhall',
    'makurmaker1',
    'jojoreginaofficial',
    'wonderversechicago'
]


json_files = [file for file in os.listdir(electronics_dir) if file.endswith('.json')]

required_features = ['ownerUsername', 'url', 'caption', 'hashtags', 'mentions', 'commentsCount', 'likesCount', 'timestamp']

for file_name in json_files:
    file_path = os.path.join(electronics_dir, file_name)

    df = pd.read_json(file_path)

    filter_df = df[df['ownerUsername'].isin(automobile_influencers)][required_features].sort_values(by='ownerUsername')

    try:
        excel_file_path = electronics_dir + '/' + file_name + '.xlsx'
        filter_df.to_excel(excel_file_path, index=False)
        print(f"DataFrame saved as {excel_file_path}")
    except Exception as e:
        print(f"Error saving DataFrame: {e}")