import threading

from thinking_buildings.events import Detection, EventBus


class TestDetection:
    def test_required_fields(self):
        det = Detection(label="cat", confidence=0.9, bbox=(0, 0, 50, 50), timestamp=1.0)
        assert det.label == "cat"
        assert det.confidence == 0.9
        assert det.bbox == (0, 0, 50, 50)
        assert det.timestamp == 1.0

    def test_optional_fields_default_none(self):
        det = Detection(label="dog", confidence=0.8, bbox=(0, 0, 10, 10), timestamp=1.0)
        assert det.identity is None
        assert det.face_confidence is None
        assert det.face_bbox is None
        assert det.occlusion is False

    def test_face_fields(self, make_detection):
        det = make_detection(
            identity="john",
            face_confidence=0.92,
            face_bbox=(110, 110, 150, 170),
            occlusion=False,
        )
        assert det.identity == "john"
        assert det.face_confidence == 0.92
        assert det.face_bbox == (110, 110, 150, 170)


class TestEventBus:
    def test_subscribe_adds_callback(self):
        bus = EventBus()
        bus.subscribe(lambda dets: None)
        assert len(bus._subscribers) == 1

    def test_publish_calls_all_subscribers(self, make_detection):
        bus = EventBus()
        calls_a, calls_b = [], []
        bus.subscribe(lambda dets: calls_a.append(dets))
        bus.subscribe(lambda dets: calls_b.append(dets))

        detections = [make_detection()]
        bus.publish(detections)

        assert len(calls_a) == 1
        assert len(calls_b) == 1
        assert calls_a[0] is detections

    def test_publish_no_subscribers_is_safe(self, make_detection):
        bus = EventBus()
        bus.publish([make_detection()])  # should not raise

    def test_callback_receives_correct_list(self, make_detection):
        bus = EventBus()
        received = []
        bus.subscribe(lambda dets: received.extend(dets))

        d1 = make_detection(label="cat")
        d2 = make_detection(label="dog")
        bus.publish([d1, d2])

        assert len(received) == 2
        assert received[0].label == "cat"
        assert received[1].label == "dog"

    def test_thread_safe_subscribe_and_publish(self, make_detection):
        """Concurrent subscribes and publishes should not raise."""
        bus = EventBus()
        results = []
        errors = []

        def subscriber(dets):
            results.extend(dets)

        def add_subscribers():
            for _ in range(50):
                bus.subscribe(subscriber)

        def do_publishes():
            for _ in range(50):
                try:
                    bus.publish([make_detection()])
                except Exception as e:
                    errors.append(e)

        t1 = threading.Thread(target=add_subscribers)
        t2 = threading.Thread(target=do_publishes)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0
