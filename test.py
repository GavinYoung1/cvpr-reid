import time
import logging
import mindspore as ms
import mindspore.ops as F
from mindspore import ops
from tools.eval_metrics import evaluate, evaluate_with_clothes


def extract_img_feature(model, dataloader):
    features, pids, camids, clothes_ids = [], ms.tensor([]), ms.tensor([]), ms.tensor([])
    for batch_idx, (imgs, batch_pids, batch_camids, batch_clothes_ids, batch_img_path) in enumerate(dataloader):
        flip_imgs = ops.flip(imgs, [3])
        _, batch_features = model(imgs)
        _, batch_features_flip = model(flip_imgs)
        batch_features += batch_features_flip
        batch_features = F.normalize(batch_features, p=2, dim=1)

        features.append(batch_features.cpu())
        pids = ops.cat((pids, batch_pids.cpu()), dim=0)
        camids = ops.cat((camids, batch_camids.cpu()), dim=0)
        clothes_ids = ops.cat((clothes_ids, batch_clothes_ids.cpu()), dim=0)
    features = ops.cat(features, 0)

    return features, pids, camids, clothes_ids


def test(config, model, queryloader, galleryloader, dataset):
    logger = logging.getLogger('reid.test')
    since = time.time()
    # Extract features 
    qf, q_pids, q_camids, q_clothes_ids = extract_img_feature(model, queryloader)
    gf, g_pids, g_camids, g_clothes_ids = extract_img_feature(model, galleryloader)
    time_elapsed = time.time() - since
    
    logger.info("Extracted features for query set, obtained {} matrix".format(qf.shape))    
    logger.info("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    since = time.time()
    m, n = qf.size(0), gf.size(0)
    distmat = ops.zeros((m,n))
    # Cosine similarity
    for i in range(m):
        distmat[i] = (- ops.mm(qf[i:i+1], gf.t())).cpu()
    distmat = distmat.numpy()
    q_pids, q_camids, q_clothes_ids = q_pids.numpy(), q_camids.numpy(), q_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()
    time_elapsed = time.time() - since
    logger.info('Distance computing in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    since = time.time()
    logger.info("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")
    time_elapsed = time.time() - since
    logger.info('Using {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    logger.info("Computing CMC and mAP only for the same clothes setting")
    cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='SC')
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    logger.info("Computing CMC and mAP only for clothes-changing")
    cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='CC')
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    return cmc[0]


def test_prcc(model, queryloader_same, queryloader_diff, galleryloader, dataset):
    logger = logging.getLogger('reid.test')
    since = time.time()
    # Extract features for query set
    qsf, qs_pids, qs_camids, qs_clothes_ids = extract_img_feature(model, queryloader_same)
    qdf, qd_pids, qd_camids, qd_clothes_ids = extract_img_feature(model, queryloader_diff)
    # Extract features for gallery set
    gf, g_pids, g_camids, g_clothes_ids = extract_img_feature(model, galleryloader)
    time_elapsed = time.time() - since
    
    logger.info("Extracted features for query set (with same clothes), obtained {} matrix".format(qsf.shape))
    logger.info("Extracted features for query set (with different clothes), obtained {} matrix".format(qdf.shape))
    logger.info("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    m, n, k = qsf.size(0), qdf.size(0), gf.size(0)
    distmat_same = ops.zeros((m, k))
    distmat_diff = ops.zeros((n, k))
    # Cosine similarity
    for i in range(m):
        distmat_same[i] = (- ops.mm(qsf[i:i+1], gf.t())).cpu()
    for i in range(n):
        distmat_diff[i] = (- ops.mm(qdf[i:i+1], gf.t())).cpu()
    distmat_same = distmat_same.numpy()
    distmat_diff = distmat_diff.numpy()
    qs_pids, qs_camids, qs_clothes_ids = qs_pids.numpy(), qs_camids.numpy(), qs_clothes_ids.numpy()
    qd_pids, qd_camids, qd_clothes_ids = qd_pids.numpy(), qd_camids.numpy(), qd_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()

    logger.info("Computing CMC and mAP for the same clothes setting")
    cmc, mAP = evaluate(distmat_same, qs_pids, g_pids, qs_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    logger.info("Computing CMC and mAP only for clothes changing")
    cmc, mAP = evaluate(distmat_diff, qd_pids, g_pids, qd_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    return cmc[0]