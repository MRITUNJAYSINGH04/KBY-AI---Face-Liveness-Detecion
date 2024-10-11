import gradio as gr
import requests
import datadog_api_client
from PIL import Image

def check_liveness(frame):
    url = "http://127.0.0.1:8080/check_liveness"
    file = {'file': open(frame, 'rb')}

    r = requests.post(url=url, files=file)
    result = r.json().get('face_state').get('result')

    html = None
    faces = None
    if r.json().get('face_state').get('is_not_front') is not None:
        liveness_score = r.json().get('face_state').get('liveness_score')
        eye_closed = r.json().get('face_state').get('eye_closed')
        is_boundary_face = r.json().get('face_state').get('is_boundary_face')
        is_not_front = r.json().get('face_state').get('is_not_front')
        is_occluded = r.json().get('face_state').get('is_occluded')
        is_small = r.json().get('face_state').get('is_small')
        luminance = r.json().get('face_state').get('luminance')
        mouth_opened = r.json().get('face_state').get('mouth_opened')
        quality = r.json().get('face_state').get('quality')

        html = ("<table>"
                    "<tr>"
                        "<th>Face State</th>"
                        "<th>Value</th>"
                    "</tr>"
                    "<tr>"
                        "<td>Result</td>"
                        "<td>{result}</td>"
                    "</tr>"
                    "<tr>"
                        "<td>Liveness Score</td>"
                        "<td>{liveness_score}</td>"
                    "</tr>"
                    "<tr>"
                        "<td>Quality</td>"
                        "<td>{quality}</td>"
                    "</tr>"
                    "<tr>"
                        "<td>Luminance</td>"
                        "<td>{luminance}</td>"
                    "</tr>"
                    "<tr>"
                        "<td>Is Small</td>"
                        "<td>{is_small}</td>"
                    "</tr>"
                    "<tr>"
                        "<td>Is Boundary</td>"
                        "<td>{is_boundary_face}</td>"
                    "</tr>"
                    "<tr>"
                        "<td>Is Not Front</td>"
                        "<td>{is_not_front}</td>"
                    "</tr>"
                    "<tr>"
                        "<td>Face Occluded</td>"
                        "<td>{is_occluded}</td>"
                    "</tr>"
                    "<tr>"
                        "<td>Eye Closed</td>"
                        "<td>{eye_closed}</td>"
                    "</tr>"
                    "<tr>"
                        "<td>Mouth Opened</td>"
                        "<td>{mouth_opened}</td>"
                    "</tr>"
                    "</table>".format(liveness_score=liveness_score, quality=quality, luminance=luminance, is_small=is_small, is_boundary_face=is_boundary_face,
                                      is_not_front=is_not_front, is_occluded=is_occluded, eye_closed=eye_closed, mouth_opened=mouth_opened, result=result))

    else:
        html = ("<table>"
            "<tr>"
                "<th>Face State</th>"
                "<th>Value</th>"
            "</tr>"
            "<tr>"
                "<td>Result</td>"
                "<td>{result}</td>"
            "</tr>"
            "</table>".format(result=result))

    try:
        image = Image.open(frame)        

        for face in r.json().get('faces'):
            x1 = face.get('x1')
            y1 = face.get('y1')
            x2 = face.get('x2')
            y2 = face.get('y2')

            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 >= image.width:
                x2 = image.width - 1
            if y2 >= image.height:
                y2 = image.height - 1

            face_image = image.crop((x1, y1, x2, y2))
            face_image_ratio = face_image.width / float(face_image.height)
            resized_w = int(face_image_ratio * 150)
            resized_h = 150

            face_image = face_image.resize((int(resized_w), int(resized_h)))

            if faces is None:
                faces = face_image
            else:
                new_image = Image.new('RGB',(faces.width + face_image.width + 10, 150), (80,80,80))

                new_image.paste(faces,(0,0))
                new_image.paste(face_image,(faces.width + 10, 0))
                faces = new_image.copy()
    except:
        pass

    return [faces, html]

with gr.Blocks() as demo:
    gr.Markdown(
        """
    # KBY-AI - Face Liveness Detecion
    We offer SDKs for face recognition, liveness detection(anti-spoofing) and ID card recognition.
    We also specialize in providing outsourcing services with a variety of technical stacks like AI(Computer Vision/Machine Learning), Mobile apps, and web apps.
    
    ##### KYC Verification Demo - https://github.com/kby-ai/KYC-Verification-Demo-Android
    ##### ID Capture Web Demo - https://id-document-recognition-react-alpha.vercel.app
    ##### Documentation - Help Center - https://docs.kby-ai.com
    """
    )
    with gr.TabItem("Face Liveness Detection"):
        gr.Markdown(
            """
        ##### Docker Hub - https://hub.docker.com/r/kbyai/face-liveness-detection
        ```bash
        sudo docker pull kbyai/face-liveness-detection:latest
        sudo docker run -e LICENSE="xxxxx" -p 8080:8080 -p 9000:9000 kbyai/face-liveness-detection:latest
        ```
        """
        )
        with gr.Row():
            with gr.Column():
                live_image_input = gr.Image(type='filepath')
                gr.Examples(['live_examples/1.jpg', 'live_examples/2.jpg', 'live_examples/3.jpg', 'live_examples/4.jpg'], 
                            inputs=live_image_input)
                check_liveness_button = gr.Button("Check Liveness")
            with gr.Column():
                liveness_face_output = gr.Image(type="pil").style(height=150)
                livness_result_output = gr.HTML()
        
        check_liveness_button.click(check_liveness, inputs=live_image_input, outputs=[liveness_face_output, livness_result_output])
    gr.HTML('<a href="https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fhuggingface.co%2Fspaces%2Fkby-ai%2FFaceLivenessDetection&countColor=%23263759"><img src="https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fhuggingface.co%2Fspaces%2Fkby-ai%2FFaceLivenessDetection&countColor=%23263759&label=VISITORS&countColor=%23263759" /></a>')        

demo.launch(server_name="0.0.0.0", server_port=7860)