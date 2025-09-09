



import pystac

catalog = pystac.Catalog.from_file('https://s3.waw3-1.cloudferro.com/swift/v1/gisat-archive/SAR/21LYG/21LYG_VV_catalog.json')
catalog.validate_all()
print("Catalog validated successfully!")