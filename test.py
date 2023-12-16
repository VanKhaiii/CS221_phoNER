import py_vncorenlp

rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='D:\\Nam3\\NLP\\VnCoreNLP')
text = "Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."
output = rdrsegmenter.word_segment(text)
print(output)
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"])