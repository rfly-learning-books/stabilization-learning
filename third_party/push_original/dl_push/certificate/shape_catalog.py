"""Twenty distinct source geometries; L/Z remain sealed-test shapes."""
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.affinity import translate
from .assets import load_geometry,certificate_polygon

NATIVE=('T','Square','RectWide','Plus','H','L','Z','C_wide','C_narrow')

def catalog():
    shapes={name:certificate_polygon(load_geometry('assets/native_v1',name)) for name in NATIVE[:5]}
    shapes.update({
      # The old C_narrow raster asset reconstructs to a solid Square. Use an
      # explicit C polygon in this new catalog; leave the old asset untouched.
      'C_narrow':unary_union([box(-.075,-.08,-.035,.08),box(-.035,.02,.075,.08),box(-.035,-.08,.075,-.02)]),
      'C_wide':unary_union([box(-.08,-.08,-.035,.08),box(-.035,.045,.08,.08),box(-.035,-.08,.08,-.045)]),
      'L':unary_union([box(-.08,-.08,-.03,.08),box(-.03,-.08,.08,-.03)]),
      'Z':unary_union([box(-.08,.035,.045,.075),box(-.025,-.035,.025,.035),box(-.045,-.075,.08,-.035)]),
      'TallRect':box(-.025,-.09,.025,.09),
      'IShape':unary_union([box(-.065,.04,.065,.075),box(-.025,-.04,.025,.04),box(-.065,-.075,.065,-.04)]),
      'UShape':unary_union([box(-.08,-.08,-.04,.08),box(.04,-.08,.08,.08),box(-.04,-.08,.04,-.04)]),
      'StepShape':Polygon([(-.08,-.07),(.08,-.07),(.08,-.025),(.035,-.025),(.035,.025),(-.015,.025),(-.015,.07),(-.08,.07)]),
      'AsymT':unary_union([box(-.08,-.065,.08,-.02),box(-.005,-.02,.04,.09)]),
      'Triangle':Polygon([(-.08,-.06),(.08,-.06),(0,.09)]),
      'Trapezoid':Polygon([(-.085,-.06),(.085,-.06),(.045,.06),(-.045,.06)]),
      'Parallelogram':Polygon([(-.085,-.05),(.045,-.05),(.085,.05),(-.045,.05)]),
      'Pentagon':Polygon([(-.07,-.06),(.07,-.06),(.08,.025),(0,.08),(-.08,.025)]),
      'Hexagon':Polygon([(-.08,0),(-.045,-.065),(.045,-.065),(.08,0),(.045,.065),(-.045,.065)]),
      'ChamferRect':Polygon([(-.07,-.055),(.07,-.055),(.09,-.035),(.09,.035),(.07,.055),(-.07,.055),(-.09,.035),(-.09,-.035)])})
    return {name:translate(p,xoff=-p.centroid.x,yoff=-p.centroid.y) for name,p in shapes.items()}

def split(name):
    return 'sealed_test' if name in ('L','Z') else ('validation' if name in ('C_wide','C_narrow') else 'development')
