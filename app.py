from paddleocr import PaddleOCRVL
import time

start = time.time()
pipeline = PaddleOCRVL()
output = pipeline.predict("ocr-test.png")
for res in output:
    res.print()
    res.save_to_json(save_path="output")
    res.save_to_markdown(save_path="output")

end = time.time()
print(f"Time taken: {end - start:.2f} seconds")
