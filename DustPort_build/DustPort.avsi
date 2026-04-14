###############################################################################
# DustPort.avsi
#
# AviSynth+ wrappers for DustPort.dll
#
# GitHub: https://github.com/schpuppa-art/DustPort
# doom9:  <TBD>
###############################################################################


###############################################################################
# DustDenoise — convenience wrapper that builds MVTools MC clips and invokes
# PixieDustPort automatically. Good default for quick script integration.
#
# Parameters
# ----------
#   clip         : source clip (YUY2 or planar YUV 8..16 bit)
#   limit        : limit-clipping bound (default 5, 8-bit scale)
#   limit_edge   : per-pixel limit on edges (-1 = disable edge-aware, default)
#   edge_thresh  : Sobel gradient threshold (default 32)
#   tr           : temporal radius 1..5 (default 1)
#   temporal     : 0=mean, 1=median (default 0)
#   planes       : "y" / "u" / "v" / combos / "yuv" (default "yuv")
#   thSAD        : MCompensate thSAD (default 300)
#   blksize      : MAnalyse block size (default 8)
#   overlap      : MAnalyse overlap (default 4)
#   truemotion   : MAnalyse truemotion flag (default true)
#
# Example
# -------
#   src = LWLibavVideoSource("clip.mkv").ConvertBits(16)
#   clean = src.DustDenoise(limit=16, limit_edge=3, edge_thresh=24, \
#                           tr=2, temporal=1, planes="y")
#
# If you need to reuse MC clips across multiple calls, build them manually and
# pass prev/next arrays to PixieDustPort directly — this wrapper rebuilds MC
# every call.
###############################################################################

function DustDenoise(clip c, int "limit", int "limit_edge", int "edge_thresh", \
                     int "tr", int "temporal", string "planes", \
                     int "thSAD", int "blksize", int "overlap", \
                     bool "truemotion")
{
    lim    = Default(limit,        5)
    ledge  = Default(limit_edge,  -1)
    ethr   = Default(edge_thresh, 32)
    tR     = Default(tr,           1)
    temp   = Default(temporal,     0)
    pl     = Default(planes,   "yuv")
    tsad   = Default(thSAD,      300)
    blk    = Default(blksize,      8)
    ov     = Default(overlap,      4)
    tm     = Default(truemotion, true)

    Assert(tR >= 1 && tR <= 5, "DustDenoise: tr must be in 1..5")

    sup = c.MSuper(pel=2, sharp=2, hpad=16, vpad=16)

    # Build MC for each delta 1..tR. AviSynth doesn't have loops so we do
    # the manual dispatch up to tr=5.
    bv1 = (tR >= 1) ? sup.MAnalyse(isb=true,  delta=1, blksize=blk, overlap=ov, search=4, truemotion=tm) : nop()
    bv2 = (tR >= 2) ? sup.MAnalyse(isb=true,  delta=2, blksize=blk, overlap=ov, search=4, truemotion=tm) : nop()
    bv3 = (tR >= 3) ? sup.MAnalyse(isb=true,  delta=3, blksize=blk, overlap=ov, search=4, truemotion=tm) : nop()
    bv4 = (tR >= 4) ? sup.MAnalyse(isb=true,  delta=4, blksize=blk, overlap=ov, search=4, truemotion=tm) : nop()
    bv5 = (tR >= 5) ? sup.MAnalyse(isb=true,  delta=5, blksize=blk, overlap=ov, search=4, truemotion=tm) : nop()
    fv1 = (tR >= 1) ? sup.MAnalyse(isb=false, delta=1, blksize=blk, overlap=ov, search=4, truemotion=tm) : nop()
    fv2 = (tR >= 2) ? sup.MAnalyse(isb=false, delta=2, blksize=blk, overlap=ov, search=4, truemotion=tm) : nop()
    fv3 = (tR >= 3) ? sup.MAnalyse(isb=false, delta=3, blksize=blk, overlap=ov, search=4, truemotion=tm) : nop()
    fv4 = (tR >= 4) ? sup.MAnalyse(isb=false, delta=4, blksize=blk, overlap=ov, search=4, truemotion=tm) : nop()
    fv5 = (tR >= 5) ? sup.MAnalyse(isb=false, delta=5, blksize=blk, overlap=ov, search=4, truemotion=tm) : nop()

    p1 = (tR >= 1) ? c.MCompensate(sup, bv1, thSAD=tsad) : nop()
    p2 = (tR >= 2) ? c.MCompensate(sup, bv2, thSAD=tsad) : nop()
    p3 = (tR >= 3) ? c.MCompensate(sup, bv3, thSAD=tsad) : nop()
    p4 = (tR >= 4) ? c.MCompensate(sup, bv4, thSAD=tsad) : nop()
    p5 = (tR >= 5) ? c.MCompensate(sup, bv5, thSAD=tsad) : nop()
    n1 = (tR >= 1) ? c.MCompensate(sup, fv1, thSAD=tsad) : nop()
    n2 = (tR >= 2) ? c.MCompensate(sup, fv2, thSAD=tsad) : nop()
    n3 = (tR >= 3) ? c.MCompensate(sup, fv3, thSAD=tsad) : nop()
    n4 = (tR >= 4) ? c.MCompensate(sup, fv4, thSAD=tsad) : nop()
    n5 = (tR >= 5) ? c.MCompensate(sup, fv5, thSAD=tsad) : nop()

    return (tR == 1) ? c.PixieDustPort(limit=lim, limit_edge=ledge, edge_thresh=ethr, \
                                        tr=1, temporal=temp, planes=pl, \
                                        prev=[p1], next=[n1]) \
         : (tR == 2) ? c.PixieDustPort(limit=lim, limit_edge=ledge, edge_thresh=ethr, \
                                        tr=2, temporal=temp, planes=pl, \
                                        prev=[p1,p2], next=[n1,n2]) \
         : (tR == 3) ? c.PixieDustPort(limit=lim, limit_edge=ledge, edge_thresh=ethr, \
                                        tr=3, temporal=temp, planes=pl, \
                                        prev=[p1,p2,p3], next=[n1,n2,n3]) \
         : (tR == 4) ? c.PixieDustPort(limit=lim, limit_edge=ledge, edge_thresh=ethr, \
                                        tr=4, temporal=temp, planes=pl, \
                                        prev=[p1,p2,p3,p4], next=[n1,n2,n3,n4]) \
         :             c.PixieDustPort(limit=lim, limit_edge=ledge, edge_thresh=ethr, \
                                        tr=5, temporal=temp, planes=pl, \
                                        prev=[p1,p2,p3,p4,p5], next=[n1,n2,n3,n4,n5])
}
