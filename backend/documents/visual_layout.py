"""Geometry-only layout hints, never semantic figure labels or official gold."""


def compact_visual_metadata(metadata, *, include_layout=False):
    metadata = dict(metadata or {})
    clean={k:v for k,v in metadata.items() if not k.startswith('raster_')}
    if include_layout and metadata.get('raster_layout_groups'):
        layout=metadata['raster_layout_groups']
        clean['native_layout_hint']={'groups':layout['groups_left_to_right'],
                                     'not_logical_figure_gold':True,
                                     'excluded_objects':layout['excluded_object_count']}
    return clean


def raster_layout_groups(boxes):
    # Ignore page-sized scan tiles, tiny marks and very long strips for this
    # OPTIONAL panel-layout hint. All original boxes remain in metadata.
    eligible = []
    for box in boxes:
        x0,y0,x1,y1 = box
        w,h=x1-x0,y1-y0
        if w <= 0 or h <= 0: continue
        if .005 <= w*h <= .9 and max(w/h,h/w) <= 8:
            eligible.append(box)

    def cluster(items, axis):
        groups=[]
        for item in sorted(items,key=lambda b:b[axis]):
            if groups and item[axis] < max(b[axis+2] for b in groups[-1]):
                groups[-1].append(item)
            else:
                groups.append([item])
        return groups

    result=[]
    for column in cluster(eligible,0):
        rows=cluster(column,1)
        result.append({'bbox':[min(b[0] for b in column),min(b[1] for b in column),
                               max(b[2] for b in column),max(b[3] for b in column)],
                       'row_raster_counts':[len(row) for row in rows],
                       'native_raster_count':len(column)})
    return {'groups_left_to_right':result,'excluded_object_count':len(boxes)-len(eligible),
            'coverage_complete':False,
            'policy':'Geometric hints only: row counts count native raster objects, not logical figures. '
                     'Tiny marks, long strips and page-sized scan tiles excluded; vectors not counted. '
                     'Validate requested region against the actual image; never use whole-page totals for a subregion.'}
