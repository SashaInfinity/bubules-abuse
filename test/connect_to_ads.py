from selenium import webdriver
from selenium.webdriver.chrome.options import Options
# import requests
# url = "http://local.adspower.com:50325/api/v1/browser/start"
# params = {
#     "user_id": "j6bxoyd"
# }

# response = requests.get(url, params=params)

# print(response.text)

chrome_driver = "C:\\Users\\Sasha\\AppData\\Roaming\\adspower_global\\cwd_global\\chrome_108\\chromedriver.exe"
chrome_options = Options()
chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:62547")
driver = webdriver.Chrome(chrome_driver, options=chrome_options)
print(driver.title)
driver.get("https://www.baidu.com")