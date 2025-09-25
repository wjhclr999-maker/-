import requests

cookies = {
    'buvid3': 'A4DD9CCB-7918-6500-039F-63EF9C76EC2260844infoc',
    'b_nut': '1753841660',
    '_uuid': 'CE2ADC4F-1578-A596-810106-D8AECC5DCAAA60412infoc',
    'enable_web_push': 'DISABLE',
    'buvid4': '8535D7B7-01E3-DC8C-E5EB-9C9DEBE671A961253-025073010-gB28yMJcp0N2UOtzGIdHWVqnzHcxVTGs9HvdfLJa5bkalUFA0yr8xmaGwuuIJmON',
    'buvid_fp': '49c6d4c8ec8f9d54c26cc81bc1151a64',
    'CURRENT_FNVAL': '2000',
    'SESSDATA': '19f313b5%2C1769394015%2C4d9d4%2A71CjBzM5Gv_nPbDueKmH22fAt6jI5-ONlVx7hqQDEKssiTjp0IrTL0DPMTixhvSPG3Ra4SVjgxQW1ucmJIMV9TX25tcDNRZllMNHVOckZxVldjc0gxdUZBWlhITXVNSWFVVV8tcDJSaUlfQTlRV3U2MXNCeWN5YUNVcVNfRWtTa3VoRUhudm1IQzdRIIEC',
    'bili_jct': '152295a243b5fd57ed3b5d0575832cbf',
    'DedeUserID': '1403417340',
    'DedeUserID__ckMd5': '0ab7a92d1b8eacd1',
    'sid': '890usjhu',
    'theme-tip-show': 'SHOWED',
    'bp_t_offset_1403417340': '1095232251211284480',
    'b_lsid': '106108610E5_198D08D3434',
    'bsource': 'search_bing',
    'bili_ticket': 'eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NTYxMDQ3NzIsImlhdCI6MTc1NTg0NTUxMiwicGx0IjotMX0.opMzGMVt-mzn8yU9nU99qzYvSzPgfH5J98CpbEZjYLQ',
    'bili_ticket_expires': '1756104712',
    'home_feed_column': '4',
    'browser_resolution': '800-725',
    'theme-avatar-tip-show': 'SHOWED',
}

headers = {
    'accept': '*/*',
    'accept-language': 'zh-CN,zh;q=0.9',
    'origin': 'https://www.bilibili.com',
    'priority': 'u=1, i',
    'referer': 'https://www.bilibili.com/',
    'sec-ch-ua': '"Not;A=Brand";v="99", "Google Chrome";v="139", "Chromium";v="139"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-site',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36',
    # 'cookie': 'buvid3=A4DD9CCB-7918-6500-039F-63EF9C76EC2260844infoc; b_nut=1753841660; _uuid=CE2ADC4F-1578-A596-810106-D8AECC5DCAAA60412infoc; enable_web_push=DISABLE; buvid4=8535D7B7-01E3-DC8C-E5EB-9C9DEBE671A961253-025073010-gB28yMJcp0N2UOtzGIdHWVqnzHcxVTGs9HvdfLJa5bkalUFA0yr8xmaGwuuIJmON; buvid_fp=49c6d4c8ec8f9d54c26cc81bc1151a64; CURRENT_FNVAL=2000; SESSDATA=19f313b5%2C1769394015%2C4d9d4%2A71CjBzM5Gv_nPbDueKmH22fAt6jI5-ONlVx7hqQDEKssiTjp0IrTL0DPMTixhvSPG3Ra4SVjgxQW1ucmJIMV9TX25tcDNRZllMNHVOckZxVldjc0gxdUZBWlhITXVNSWFVVV8tcDJSaUlfQTlRV3U2MXNCeWN5YUNVcVNfRWtTa3VoRUhudm1IQzdRIIEC; bili_jct=152295a243b5fd57ed3b5d0575832cbf; DedeUserID=1403417340; DedeUserID__ckMd5=0ab7a92d1b8eacd1; sid=890usjhu; theme-tip-show=SHOWED; bp_t_offset_1403417340=1095232251211284480; b_lsid=106108610E5_198D08D3434; bsource=search_bing; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NTYxMDQ3NzIsImlhdCI6MTc1NTg0NTUxMiwicGx0IjotMX0.opMzGMVt-mzn8yU9nU99qzYvSzPgfH5J98CpbEZjYLQ; bili_ticket_expires=1756104712; home_feed_column=4; browser_resolution=800-725; theme-avatar-tip-show=SHOWED',
}

params = {
    'display_id': '1',
    'request_cnt': '15',
    'from_region': '1003',
    'device': 'web',
    'plat': '30',
    'web_location': '333.40138',
    'w_rid': '8ac5bbc477a80858655820331a05039e',
    'wts': '1755849373',
}

response = requests.get(
    'https://api.bilibili.com/x/web-interface/region/feed/rcmd',
    params=params,
    cookies=cookies,
    headers=headers,
)


import pandas as pd

data = response.json()['data']['archives']
df = pd.DataFrame(data)

# 选择需要的列并转为中文列名,即bvid，title，duration，pubdate转为中文
df = df[['bvid', 'title', 'duration', 'pubdate']]
df['pubdate'] = pd.to_datetime(df['pubdate'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
df['duration'] = df['duration'].apply(lambda x: f"{x // 60}分{x % 60}秒")
df['bvid'] = df['bvid'].apply(lambda x: f"https://www.bilibili.com/video/{x}")

df.to_excel('bilibili.xlsx',
             index=False,
             header=['BVID','标题','时长','发布时间 '],
             engine='openpyxl')