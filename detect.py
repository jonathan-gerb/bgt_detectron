from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os


def setup_dataset(name, basedir):
    register_coco_instances(name, {},
                            os.path.join(basedir, "annotation.json"),
                            os.path.join(basedir, "train/"))

    metadata = MetadataCatalog.get(name)
    dataset_dicts = DatasetCatalog.get(name)
    return metadata, dataset_dicts


def setup_config(args):
    cfg = get_cfg()
    cfg.merge_from_file(
        "./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.DATASETS.TRAIN = (args.dataset_name)
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.MODEL.WEIGHTS = args.model_weights  # initialize from model zoo
    # cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = args.images_per_batch  # 20
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_threshold
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        args.roi_batch_size_per_image
    )  # faster, and good enough for this toy dataset

    return config


def detect(predictor, metadata, dataset_dicts, basedir):
    filepaths = glob.glob(basedir + "*.png")
    newfile = pygeoj.new()
    newfile.define_crs(type="link", link="https://spatialreference.org/ref/epsg/28992/esriwkt/", link_type="esriwkt")

    for filepath in filepaths:
        print("segmenting: ", filepath)
        coords = os.path.basename(filepath).split(".png")[0].split("_")[:2]
        coords = [float(c) for c in coords]

        x = coords[0]
        y = coords[1]

        # make transform
        transform = Affine.translation(float(x), (float(y) + 50)) * Affine.scale(0.1, -0.1)

        im = cv2.imread(filepath)
        outputs = predictor(im)
        bboxes = np.array(outputs['instances'].pred_boxes.tensor.cpu())
        masks = np.array(outputs['instances'].pred_masks.cpu())

        for i in range(bboxes.shape[0]):
            polygon = Mask(masks[i]).polygons()
            points = polygon.points
            poly_list = []
            if len(points) == 0:
                continue

            for pair in points[0]:
                p_x, p_y = transform * (pair[0], pair[1])
                poly_list.append((p_x, p_y))

            newfile.add_feature(properties={"class": "woonboot"},
                                geometry={"type": "Polygon", "coordinates": [poly_list]})

    newfile.add_all_bboxes()
    newfile.update_bbox()
    newfile.add_unique_id()
    newfile.save("./all_objects_test.geojson")


def run():
    metadata, dataset_dicts = setup_dataset(args.dataset_name, args.basedir)
    cfg = setup_config(args)
    predictor = DefaultPredictor(cfg)
    detect(predictor, metadata, dataset_dicts, args.basedir)


if __name__ == '__main__':
    parser.add_argument('--dataset-name', type=str, required=True,
                        help="name of dataset")
    parser.add_argument('--num-workers', type=int, default=5,
                        help="number of workers for dataloader")
    parser.add_argument('--basedir', type=str, required=True,
                        help="base directory of dataset")
    parser.add_argument('--images-per-batch', type=str, default=20,
                        help="images per batch")
    parser.add_argument('--model-weights', type=str, required=True,
                        help="path to model.pth file, i.e. /training/model_final.pth")
    parser.add_argument('--score-threshold', type=str, default=0.7,
                        help="minimum score for detection")
    parser.add_argument('--roi-batch-size-per-image', type=str, default=512,
                        help="regions of interest to generate per image")
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    run(args)
