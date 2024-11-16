import time

from openai import OpenAI


from common.log import logger
from common.token_bucket import TokenBucket
from config import conf


# OPENAI提供的画图接口
class OpenAIImage(object):
    def __init__(self):
        self.client = OpenAI(api_key=conf().get("open_ai_api_key"))
        if conf().get("rate_limit_dalle"):
            self.tb4dalle = TokenBucket(conf().get("rate_limit_dalle", 50))

    def create_img(self, query, retry_count=0, api_key=None, api_base=None):
        try:
            if conf().get("rate_limit_dalle") and not self.tb4dalle.get_token():
                return False, "请求太快了，请休息一下再问我吧"
            logger.info("[OPEN_AI] image_query={}".format(query))
            response = self.client.images.generate(api_key=api_key,
            prompt=query,  # 图片描述
            n=1,  # 每次生成图片的数量
            model=conf().get("text_to_image") or "dall-e-2")
            image_url = response.data[0].url
            logger.info("[OPEN_AI] image_url={}".format(image_url))
            return True, image_url
        except openai.RateLimitError as e:
            logger.warn(e)
            if retry_count < 1:
                time.sleep(5)
                logger.warn("[OPEN_AI] ImgCreate RateLimit exceed, 第{}次重试".format(retry_count + 1))
                return self.create_img(query, retry_count + 1)
            else:
                return False, "画图出现问题，请休息一下再问我吧"
        except Exception as e:
            logger.exception(e)
            return False, "画图出现问题，请休息一下再问我吧"
