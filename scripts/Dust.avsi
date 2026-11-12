###############################################################################
# Dust.avsi - AviSynth+ wrapper for the Dust plugin (Dust.dll)
#
# DustDenoise() builds motion compensated neighbour frames with MVTools2 and
# calls FaeryDust / PixieDust / GoldDust (or SpaceDust, spatial only, no MC).
#
# Needs: AviSynth+ 3.7.3 or newer (arrays, ArrayAdd, for loops), Dust.dll,
#        mvtools2.dll (pinterf) unless mc=false.
# https://github.com/Selur/Dust
###############################################################################
#
# Parameters
# ----------
#   mode        : "faery", "pixie" (default), "gold" or "space"
#   limit       : limit-clipping bound, 8-bit scale (default per mode: 2/5/8/6)
#   limit_edge  : limit on edges, -1 = edge-aware limit off (default)
#   edge_thresh : Sobel magnitude counted as full edge (default 32)
#   tr          : temporal radius 1..5 (default 1)
#   temporal    : 0 = mean (default), 1 = median
#   planes      : "y", "u", "v" or combinations (default "yuv")
#   scthresh    : scene change threshold, 0 = frame properties only, -1 = off
#                 (default 20)
#   interlaced  : true = field mode (motion compensation per field as well);
#                 set the field order first (AssumeTFF / AssumeBFF)
#   mc          : false = no motion compensation, Dust uses the plain
#                 neighbour frames (default true)
#   thSAD, blksize, overlap, search, pel, truemotion : MVTools settings
#                 (defaults 300, 8, 4, 4, 2, true)
#   opt         : instruction set, -1 = best (default)
#
# Example
# -------
#   LoadPlugin("Dust.dll")
#   LoadPlugin("mvtools2.dll")
#   Import("Dust.avsi")
#   src = LWLibavVideoSource("clip.mkv").ConvertBits(16)
#   src.DustDenoise(limit=16, limit_edge=3, edge_thresh=24, tr=2, temporal=1, planes="y")
#
# To reuse the motion compensated clips for several calls, build them once and
# pass prev/next to PixieDust directly; this wrapper rebuilds them every call.
###############################################################################

function DustDenoise(clip c, string "mode", int "limit", int "limit_edge", int "edge_thresh", \
                     int "tr", int "temporal", string "planes", int "scthresh", bool "interlaced", \
                     bool "mc", int "thSAD", int "blksize", int "overlap", int "search", int "pel", \
                     bool "truemotion", int "opt")
{
    md    = LCase(Default(mode, "pixie"))
    Assert(md == "faery" || md == "pixie" || md == "gold" || md == "space", \
           "DustDenoise: mode must be faery, pixie, gold or space")
    lim   = Default(limit, md == "faery" ? 2 : md == "pixie" ? 5 : md == "gold" ? 8 : 6)
    ledge = Default(limit_edge, -1)
    ethr  = Default(edge_thresh, 32)
    tR    = Default(tr, 1)
    temp  = Default(temporal, 0)
    pl    = Default(planes, "yuv")
    sc    = Default(scthresh, 20)
    il    = Default(interlaced, false)
    usemc = Default(mc, true)
    tsad  = Default(thSAD, 300)
    blk   = Default(blksize, 8)
    ov    = Default(overlap, 4)
    srch  = Default(search, 4)
    pl_   = Default(pel, 2)
    tm    = Default(truemotion, true)
    o     = Default(opt, -1)
    Assert(tR >= 1 && tR <= 5, "DustDenoise: tr must be 1..5")

    if (md == "space") {
        return c.SpaceDust(limit=lim, limit_edge=ledge, edge_thresh=ethr, planes=pl, scthresh=sc,                            interlaced=il, opt=o)
    }
    if (!usemc) {
        return md == "faery" ? c.FaeryDust(limit=lim, limit_edge=ledge, edge_thresh=ethr, tr=tR, temporal=temp,                                            planes=pl, scthresh=sc, interlaced=il, opt=o)              : md == "pixie" ? c.PixieDust(limit=lim, limit_edge=ledge, edge_thresh=ethr, tr=tR, temporal=temp,                                            planes=pl, scthresh=sc, interlaced=il, opt=o)              :                 c.GoldDust(limit=lim, limit_edge=ledge, edge_thresh=ethr, tr=tR, temporal=temp,                                           planes=pl, scthresh=sc, interlaced=il, opt=o)
    }

    # Motion compensation on the frames, or per field (same field of frame n +- k = field n +- 2k).
    chroma = !c.IsY() && (FindStr(LCase(pl), "u") > 0 || FindStr(LCase(pl), "v") > 0)
    src  = il ? c.SeparateFields() : c
    step = il ? 2 : 1
    sup  = src.MSuper(pel=pl_, sharp=2, hpad=16, vpad=16)
    prev = []
    next = []
    for (k = 1, tR) {
        bv = sup.MAnalyse(isb=false, delta=k * step, blksize=blk, overlap=ov, search=srch, truemotion=tm,                           chroma=chroma)
        fv = sup.MAnalyse(isb=true, delta=k * step, blksize=blk, overlap=ov, search=srch, truemotion=tm,                           chroma=chroma)
        p = src.MCompensate(sup, bv, thSAD=tsad)
        n = src.MCompensate(sup, fv, thSAD=tsad)
        prev = ArrayAdd(prev, il ? p.Weave() : p)
        next = ArrayAdd(next, il ? n.Weave() : n)
    }

    return md == "faery" ? c.FaeryDust(limit=lim, limit_edge=ledge, edge_thresh=ethr, tr=tR, temporal=temp,                                        planes=pl, scthresh=sc, interlaced=il, prev=prev, next=next, opt=o)          : md == "pixie" ? c.PixieDust(limit=lim, limit_edge=ledge, edge_thresh=ethr, tr=tR, temporal=temp,                                        planes=pl, scthresh=sc, interlaced=il, prev=prev, next=next, opt=o)          :                 c.GoldDust(limit=lim, limit_edge=ledge, edge_thresh=ethr, tr=tR, temporal=temp,                                       planes=pl, scthresh=sc, interlaced=il, prev=prev, next=next, opt=o)
}
