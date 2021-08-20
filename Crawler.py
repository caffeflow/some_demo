

'''
根据关键字获取今日头条图集
'''
 
 
import os
import requests
from urllib.parse import quote
from hashlib import md5
from multiprocessing.pool import Pool
 
baseurl = 'https://www.toutiao.com/search_content/?'
total = 1
 
 
def get_page(page, keyword):  # 获得页面
 
    keyword_h = quote(keyword)
    
    headers = {
    'referer': 'https://www.toutiao.com/search/?keyword='+keyword_h,
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/67.0.3396.99 Chrome/67.0.3396.99 Safari/537.36',
    'x-requested-with': 'XMLHttpRequest'
    }
 
    params = {
        'offset': page*20,
        'format': 'json',
        'keyword': keyword,
        'autoload': 'true',
        'count': '20',
        'cur_tab': '3',
        'from': 'gallery',
        }
        
    try:
        response = requests.get(baseurl, headers=headers, params=params)
        if response.status_code == 200:
            #print(response.url)
            return response.json()
    except ConnectionError as e:
        print('error', e.args)
      
   
def get_image(json):  # 解析json文件中的图集标题和每张图的url，并逐步yield
    items = json.get('data')
    if items:
        for item in items:
            title = item.get('title')
            image_list = item.get('image_list')
            for image in image_list:
                url = 'http:'+image.get('url').replace('list', 'origin')
                yield{
                    'url': url,
                    'title': title
                }
                
                
def save_image(image):  # 根据标题保存图片
    if not os.path.exists(image.get('title')):
        os.mkdir(image.get('title'))
    try:
        response = requests.get(image.get('url'))
        if response.status_code == 200:
            file_path = '{0}/{1}.{2}'.format(image.get('title'), md5(response.content).hexdigest(), 'jpg')
            if not os.path.exists(file_path):
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            else:
                print('Already Downloaded', file_path)
    except requests.ConnectionError:
        print('Failed to Save Image')
        
    
def main(page, keyword='可爱'):  # 主流程控制
    if not os.path.exists(keyword):
        os.mkdir(keyword)
    os.chdir(keyword)
    json = get_page(page,keyword)
    for item in get_image(json):
        save_image(item)
    os.chdir(os.pardir)
          
          
if __name__ == '__main__':  # 增加多线程 
    pool = Pool()
    groups_one = ([i for i in range(total)])
    pool.map(main, groups_one)
    pool.close()
    pool.join()
