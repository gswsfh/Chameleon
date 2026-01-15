import datetime
import threading
import time

import pyshark
import psutil
from selenium import webdriver
from selenium.webdriver import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import asyncio

asyncio.set_event_loop(asyncio.new_event_loop())
proxy_ip="localhost"
proxy_port=7897
stop_sniff = threading.Event()

def sniff_traffic():
    pcap_path="google_search_traffic.pcap"
    asyncio.set_event_loop(asyncio.new_event_loop())
    cap = pyshark.LiveCapture(interface="Adapter for loopback traffic capture",
                              bpf_filter='host 127.0.0.1 and port 7897',output_file=pcap_path)
    try:
        for pkt in cap.sniff_continuously():
            try:
                src = f"{pkt.ip.src}:{pkt[pkt.transport_layer].srcport}"
                dst = f"{pkt.ip.dst}:{pkt[pkt.transport_layer].dstport}"
                proto = pkt.transport_layer  # TCP or UDP
            except AttributeError:  
                src = dst = proto = "N/A"
            print(f"{datetime.datetime.now():%M:%S.%f}  "
                  f"{src:<22} -> {dst:<22}  {proto:<3}  {pkt.length:>4} B")
    finally:
        cap.close()
        

def browse(wordlist):

    chrome_options = Options()
    chrome_options.add_argument("--remote-allow-origins=*")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("disable-infobars")
    chrome_options.add_argument('--proxy-server={}:{}'.format(proxy_ip, proxy_port))
    driver = webdriver.Chrome(options=chrome_options)
    try:
        for word in wordlist:
            driver.get("https://www.google.com")
            search_box=driver.find_element(By.CLASS_NAME, "gLFyf")
            search_box.send_keys(word)
            search_box.send_keys(Keys.RETURN)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "gLFyf"))  
            )
            time.sleep(2)
    finally:
        driver.quit()

if __name__ == '__main__':
    wordlist=[
        "Gemini","India vs England","Charlie Kirk","Club World Cup","India vs Australia",
        "DeepSeek","Asia Cup","Iran","iPhone 17","Pakistan and India"
    ]
    t = threading.Thread(target=sniff_traffic, daemon=True)
    t.start()
    browse(wordlist)
    stop_sniff.set()
    t.join(timeout=3)  



