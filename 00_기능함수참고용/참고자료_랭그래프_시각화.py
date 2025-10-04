def langgraph_img() :
    img = input("파일이름을 입력하세요 : ")
    file_path = f"img/{img}.png"
    try:
        png_data = app.get_graph().draw_mermaid_png()
        with open(file_path, "wb") as f:
            f.write(png_data)
        print(f"그래프 이미지 저장됨: {file_path}")
    except Exception as e:
        print(f"PNG 저장 실패: {e}")
        print("Graphviz가 설치되지 않았을 수 있습니다.")