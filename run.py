dataset = KittiDatasetLoader("path/to/kitti")
sample = dataset.get_sample(0)

question = "빨간 차는 어디에 있어?"
parser = QuestionParser(question)
target = parser.extract_target_object()

detector = ObjectDetector()
objects = detector.detect_objects(sample["image"])

grounder = VisualGrounder()
matched_bbox = grounder.match_question_to_object(target, objects)

visualizer = Visualizer()
visualizer.draw_grounding_result(sample["image"], matched_bbox)
