"""Twelve extra curved outlines. Polygonal approximations, not analytic arcs."""
from shapely.geometry import Point, LineString, Polygon, box
from shapely.affinity import scale, translate
from shapely.ops import unary_union


def arc_catalog():
    def rounded(p, radius=.028):
        return p.buffer(radius, resolution=24)
    shapes = {
        'Ellipse': scale(Point(0,0).buffer(.065,resolution=32),1.4,1.),
        'Capsule': rounded(LineString([(-.045,0),(.045,0)]),.04),
        'RoundedRect': rounded(box(-.05,-.027,.05,.027),.03),
        'RoundedTriangle': rounded(Polygon([(-.055,-.035),(.055,-.035),(0,.055)])),
        'RoundedTrapezoid': rounded(Polygon([(-.055,-.035),(.055,-.035),(.025,.04),(-.025,.04)])),
        'RoundedPentagon': rounded(Polygon([(-.04,-.035),(.04,-.035),(.05,.01),(0,.055),(-.05,.01)])),
        'RoundedL': rounded(unary_union([box(-.04,-.045,-.005,.045),box(-.04,-.045,.045,-.01)])),
        'RoundedT': rounded(unary_union([box(-.05,-.04,.05,-.012),box(-.014,-.012,.014,.05)])),
        'RoundedCross': rounded(unary_union([box(-.055,-.014,.055,.014),box(-.014,-.055,.014,.055)])),
        # Thick rounded C: the thin crescent trial is archived as a failed
        # contact geometry, not shipped as a valid mode library.
        'ArcC': rounded(box(-.045,-.045,.035,.045),.03).difference(Point(.065,0).buffer(.043,resolution=24)),
        'DShape': Point(0,0).buffer(.085,resolution=32).intersection(box(-.025,-.10,.10,.10)),
        'OffsetCapsule': rounded(Polygon([(-.055,-.015),(.035,-.025),(.055,.025),(-.035,.025)]),.028),
    }
    return {n:translate(p,xoff=-p.centroid.x,yoff=-p.centroid.y) for n,p in shapes.items()}
