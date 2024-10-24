from flask import Flask, request
from diffusers import DiffusionPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler
import base64
import torch
from compel import Compel


# epiCRealismの場合
# model_id = "emilianJR/epiCRealism"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
if torch.cuda.is_available():
    pipe = pipe.to("cuda")

app = Flask(__name__)

@app.route("/create", methods=['POST'])
def create():
    data_dict = request.get_json()
    prompt = data_dict['prompt']

    # ネガティブプロンプトの追加
    negative_prompt = """
    verybadimagenegative_v1.3, ng_deepnegative_v1_75t, (ugly face:0.8),cross-eyed,sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, bad anatomy, DeepNegative, facing away, tilted head, {Multiple people}, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worstquality, low quality, normal quality, jpegartifacts, signature, watermark, username, blurry, bad feet, cropped, poorly drawn hands, poorly drawn face, mutation, deformed, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, extra fingers, fewer digits, extra limbs, extra arms,extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross-eyed,mutated hands, polar lowres, bad body, bad proportions, gross proportions, text, error, missing fingers, missing arms, missing legs, extra digit, extra arms, extra leg, extra foot, ((repeating hair)), facial distortion, facial exaggerated features, unnatural facial proportions, unnatural human facial symmetry, any facial deformities
    """
    positive_scale = 7.5  # ポジティブなプロンプトの制御強度
    negative_scale = 5.0  # ネガティブプロンプトの制御強度

    # compel使わないとinputトークンは77tokenが上限になってしまうので、compelを使います
    # コンディショニングテンソルの作成
    conditioning = compel.build_conditioning_tensor(prompt)
    negative_conditioning = compel.build_conditioning_tensor(negative_prompt)

    pipe.to("cuda")
    # ガイダンススケールを使用して、プロンプトのポジティブおよびネガティブ効果を調整
    image = pipe(prompt_embeds=conditioning, negative_prompt_embeds=negative_conditioning, width=512, height=512,
                 num_inference_steps=30, guidance_scale=positive_scale, negative_guidance_scale=negative_scale).images[0]
    image.save('tmp.png')
    with open("tmp.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    # b64エンコードの文字列を返します
    return {"b64_json": encoded_string}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
