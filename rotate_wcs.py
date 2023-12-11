#!/bin/env python

#  Rotates a simple 2D FITS image. If <xcen> and <ycen> are specified,
#  the output image is the same size as the input image. Otherwise,
#  the output image is the smallest that will hold all the rotated input
#  data.

#  Usage:
#    rotate_wcs.py <infile> <outfile> <rotation> [<xcen> <ycen>]
#
#    <infile> = Input fits file
#    <outfile> = Output fits file
#    <rotation> = angle of rotation in degrees
#    <xcen> = Optional - the X pixel at the centre of rotation
#    <ycen> = Optional - the Y pixel at the centre of rotation

#  Notes:
#     - Requires pyast version 2.3

import pyfits
import sys
import math
import starlink.Ast as Ast
import starlink.Atl as Atl

#  Open the FITS file using pyfits. A list of the HDUs in the FITS file is
#  returned.
hdu_list = pyfits.open( sys.argv[1] )

#  Create an object that will transfer FITS header cards between an AST
#  FitsChan and the PyFITS header describing the primary HDU of the
#  supplied FITS file.
adapter = Atl.PyFITSAdapter( hdu_list[ 0 ] )

#  Create a FitsChan and use the above adapter to copy all the header
#  cards into it.
fitschan = Ast.FitsChan( adapter, adapter )

#  Get the flavour of FITS-WCS used by the header cards currently in the
#  FitsChan. This is so that we can use the same flavour when we write
#  out the modified WCS.
encoding = fitschan.Encoding

#  Read WCS information from the FitsChan. Additionally, this removes all
#  WCS information from the FitsChan. The returned wcsinfo object
#  is an AST FrameSet, in which the current Frame describes WCS coordinates
#  and the base Frame describes pixel coodineates. The FrameSet includes a
#  Mapping that specifies the transformation between the two Frames.
wcsinfo = fitschan.read()

#  Check that the FITS header contained WCS in a form that can be
#  understood by AST.
if wcsinfo == None:
   print( "Failed to read WCS information from {0}".format(sys.argv[1]) )

#  Rotation is restricted to 2D arrays, so check the FITS file has 2 pixel
#  axes (given by Nin) and 2 WCS axes (given by Nout).
elif wcsinfo.Nin != 2 or wcsinfo.Nout != 2:
   print( "{0} is not 2-dimensional".format(sys.argv[1]) )

#  Proceed if the WCS information was read OK.
else:

   #  Get the lower and upper bounds of the input image in pixel indices.
   #  All FITS arrays by definition have lower pixel bounds of [1,1] (unlike
   #  NDFs). Note, unlike pyfits AST uses FITS ordering for storing pixel axis
   #  values in an array (i.e. NAXIS1 first, NAXIS2 second, etc).
   lbnd_in = [ 1, 1 ]
   ubnd_in = [ fitschan["NAXIS1"], fitschan["NAXIS2"] ]

   #  Get the rotation angle and convert from degrees to radians.
   angle = float( sys.argv[3] )*Ast.DD2R

   #  Construct a MatrixMap that rotates by the required angle, converted to radians.
   sinang = math.sin( angle )
   cosang = math.cos( angle )
   pixmap = Ast.MatrixMap( [ [cosang,sinang], [-sinang,cosang] ] )

   #  If supplied, get the centre of rotation.
   if len(sys.argv) > 5:
      x0 =  float( sys.argv[4] )
      y0 =  float( sys.argv[5] )

      #  Create aShiftMap to shift the origin and combine it with the
      #  MatrixMap, to give the desired centre of rotation (shift,
      #  rotate, then shift back again).
      shiftmap = Ast.ShiftMap( [ -x0, -y0 ] )
      pixmap = Ast.CmpMap( shiftmap, pixmap );
      shiftmap.invert()
      pixmap = Ast.CmpMap( pixmap, shiftmap );

      # The dimensions of the output array are the same as the input array.
      lbnd_out = lbnd_in
      ubnd_out = ubnd_in

   #  If no centre of rotation was specified, we use the MatrixMap
   #  unchanged, and determine the bounds of the smallest output image that
   #  will hold the entire rotated input image. These are with respect to
   #  the pixel coordinates of the input array, and so some corners may have
   #  negative pixel coordinates.
   else:
      ( lb1, ub1, xl, xu ) = pixmap.mapbox( lbnd_in, ubnd_in, 1 )
      ( lb2, ub2, xl, xu ) = pixmap.mapbox( lbnd_in, ubnd_in, 2 )
      lbnd_out = [ int(lb1), int(lb2) ]
      ubnd_out = [ int(ub1), int(ub2) ]

   #  Get the value used to represent missing pixel values
   if "BLANK" in fitschan:
      badval = fitschan["BLANK"]
      flags = Ast.USEBAD
   else:
      badval = 0
      flags = 0

   # Resample the data array using the above mapping.
   ( npix, out, out_var ) = pixmap.resample( lbnd_in, ubnd_in,
                                             hdu_list[0].data, None,
                                             Ast.LINEAR, None, flags,
                                             0.05, 1000, badval, lbnd_out,
                                             ubnd_out, lbnd_out, ubnd_out )

   #  Store the rotated data in the HDU, and update the NAXISi keywords
   #  to hold the number of pixels along each edge of the rotated image.
   hdu_list[0].data = out
   fitschan["NAXIS1"] = ubnd_out[ 0 ] - lbnd_out[ 0 ] + 1
   fitschan["NAXIS2"] = ubnd_out[ 1 ] - lbnd_out[ 1 ] + 1

   #  Move the pixel origin of the output from "lbnd_out" to (1,1). Create a
   #  suitable ShiftMap, and append it to the total "old pixel coord" to
   #  "new pixel coords" mapping.
   shiftmap = Ast.ShiftMap( [ -lbnd_out[ 0 ], -lbnd_out[ 1 ] ] )
   pixmap = Ast.CmpMap( pixmap, shiftmap )

   #  Re-map the base Frame (i.e. the pixel coordinates Frame) so that it
   #  refers to the new data grid instead of the old one.
   wcsinfo.remapframe( Ast.BASE, pixmap )

   #  Attempt to write the modified WCS information to the primary HDU (i.e.
   #  convert the FrameSet to a set of FITS header cards stored in the
   #  FITS file). Indicate that we want to use original flavour of FITS-WCS.
   fitschan.Encoding = encoding
   fitschan.clear('Card')
   if fitschan.write( wcsinfo ) == 0 :
      print( "Failed to convert the rotated WCS to Fits-WCS" )

   #  If successfull, force the FitsCHan to copy its contents into the
   #  PyFITS header, then write the changed data and header to the output
   #  FITS file.
   else:
      fitschan.writefits()
      hdu_list.writeto(sys.argv[2],clobber=True)
