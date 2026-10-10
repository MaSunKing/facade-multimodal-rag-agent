import unittest
from backend.sales.directory_semantics import locate_view, hierarchy_members, VIEWS
from backend.sales.directory_evidence import collection_packets, directory_coverage
from backend.sales.prepared_support import fit_support_budget

class DirectorySemanticsTests(unittest.TestCase):
    def test_level_metadata_survives_actual_handoff(self):
        from backend.sales.directory_evidence import collection_metadata
        from backend.sales.entity_directory import directory_context
        row={'directory_level':'family_with_requested_options',
             'directory_semantic_selection':{'views':['systems','colors']},
             'directory_members':[{'attributes':{'family':'Panel'}}],
             'entity_coverage_ids':['a','b']}
        copy={**collection_metadata(row),'entity_coverage_ids':row['entity_coverage_ids']}
        view=directory_context({'meta':{'relation_scope':{'entity_coverage':{'target_ids':['a','b']}}}},[copy])
        self.assertEqual(view['target_names'],['Panel'])
        self.assertEqual(view['target_count'],1)
        self.assertEqual(view['leaf_records_represented_count'],2)

    def test_unresolved_lookup_can_read_public_navigation(self):
        import sqlite3,tempfile
        from contextlib import closing
        from pathlib import Path
        from backend.tests.sales.test_directory_evidence import DirectoryEvidenceTests
        from backend.sales.directory_evidence import from_directory
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'test.sqlite3'
            DirectoryEvidenceTests().fixture(path)
            with closing(sqlite3.connect(path)) as c:
                c.row_factory=sqlite3.Row
                rows=from_directory(c,{},['02_product_information'],question='What is inside?',
                    operation='lookup',scorer=self.rank('cores','composition'))
            self.assertTrue(rows)
            self.assertTrue(all(r['directory_level']=='family_with_requested_options' for r in rows))
            self.assertNotIn('PRIVATE',str(rows))

    def rank(self,*wanted):
        return lambda q,descriptions:[1 if key in wanted else 0 for key in VIEWS]

    def members(self):
        return [{'id':'a','name':'Red rough','attributes':{'family':'Panel','color_name':'Red',
                 'surface_finish':'rough','available_surface_finishes':['rough','smooth']},'citations':[]},
                {'id':'b','name':'Blue smooth','attributes':{'family':'Panel','color_name':'Blue',
                 'surface_finish':'smooth'},'citations':[]},
                {'id':'c','name':'Wool composite','attributes':{'family':'Composite',
                 'insulation_core':'Wool'},'citations':[]}]

    def test_view_ranking_preserves_alternatives_not_confidence(self):
        view=locate_view('What is inside?',scorer=self.rank('cores','composition'))
        self.assertEqual(set(view['views']),{'cores','composition'})
        self.assertFalse(view['semantic_support_verified'])
        self.assertEqual(len(view['ranked_candidates']),len(VIEWS))

    def test_fallback_is_broad_navigation_not_false_support(self):
        view=locate_view('question',scorer=lambda *args:[float('nan')]*len(VIEWS))
        self.assertEqual(view['status'],'broad_navigation_fallback')
        self.assertIn('systems',view['views'])

    def test_systems_are_parents_options_are_not_skus(self):
        view=locate_view('overview',scorer=self.rank('systems','colors'))
        rows=hierarchy_members(self.members(),view,{'c':{'colors':['Red'],'cores':['Wool','Foam']}})
        self.assertEqual(len(rows),2)
        text='\n'.join(r['text'] for r in rows)
        self.assertIn('Red',text)
        self.assertIn('Foam、Wool',text)
        self.assertNotIn('Red rough',text)
        self.assertNotIn('Wool composite',text)
        self.assertEqual({i for row in rows for i in row['child_ids']},{'a','b','c'})

    def test_core_scope_does_not_dump_unrequested_colors(self):
        view=locate_view('inside',scorer=self.rank('cores','composition'))
        rows=hierarchy_members(self.members(),view,{})
        self.assertIn('Wool','\n'.join(r['text'] for r in rows))
        self.assertNotIn('Red','\n'.join(r['text'] for r in rows))

    def test_summary_none_preserves_actual_directory_membership(self):
        rows=collection_packets('02_product_information',[
            {'id':str(i),'name':'Parent '+str(i),'text':'Parent '+str(i)+'; '+'details '*60} for i in range(5)])
        counter=lambda values:sum(len(r['text']) for r in values)
        kept,audit=fit_support_budget(rows,question='overview',operation='summary',collection_action='none',
            audit_tokens=counter,generation_tokens=counter,ceiling=300)
        self.assertTrue(audit['budget_satisfied'])
        self.assertEqual(directory_coverage(rows,kept)[0]['retained_count'],5)
        self.assertTrue(audit['directory_compaction']['applied'])

if __name__=='__main__':unittest.main()
