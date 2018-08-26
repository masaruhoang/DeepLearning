# -*- coding: utf-8 -*-
import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

from test_scrapy.items import ImageItem

class SaveYahooImageSpider(CrawlSpider):
    name = 'save_yahoo_image'
    allowed_domains = ["www.s-b-c.net"]
    start_urls = ["https://www.s-b-c.net/photo/index.cgi?ss2=4&&wd=&page=2"]

    rules = (
        Rule(LinkExtractor(allow=( )), callback="parse_page", follow=True),
    )

    def parse_page(self, response):
        print("\n>>> Parse " + response.url + " <<<")
        item = ImageItem()
        item["image_directory_name"] = self.start_urls[0].rsplit("/", 1)[1]
        item["image_urls"] = []
		
        text1 = '//h3/text()'
        text2 = '美容皮膚科'
        #if(text2 in response.xpath(text1).extract()):
        for image_url in response.xpath('//div[@class="photo-contents-left-cell1"]/a[contains(@href, "photo")]/img/@src').extract():
            if "http" not in image_url:
                item["image_urls"].append(response.url.rsplit("/", 1)[0] + "/" + image_url)
            else:
                item["image_urls"].append(image_url)
					
        # print(vars(item))
        return item

