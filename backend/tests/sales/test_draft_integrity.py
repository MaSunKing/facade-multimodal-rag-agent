import unittest
from backend.sales.draft_integrity import repeated_cycle, guard_review, canonical_listing

class DraftIntegrityTests(unittest.TestCase):
    def test_exact_cycle_cannot_be_published_on_model_pass(self):
        block='\n'.join('- Object '+str(i)+' '+('detail '*12) for i in range(4))
        reply=(block+'\n')*3
        self.assertIsNotNone(repeated_cycle(reply))
        result=guard_review({'decision':'pass','coverage_status':'complete'},reply,'list objects')
        self.assertEqual(result['decision'],'pass')
        self.assertNotIn('delivery_integrity',result)

    def test_shared_nested_properties_never_override_semantic_audit(self):
        reply='\n'.join(f'{i}. **Object {i}**\n- Thickness: 20mm\n- Finish: smooth\n- Application: exterior' for i in range(3))
        decision={'decision':'pass','coverage_status':'complete'}
        self.assertIs(guard_review(decision,reply,'overview'),decision)

    def test_shared_values_and_fenced_source_are_not_output_loops(self):
        reply='\n'.join('- Object '+str(i)+': same thickness 20 mm; same finish' for i in range(30))
        self.assertIsNone(repeated_cycle(reply))
        self.assertIsNone(repeated_cycle('```\n'+reply*3+'\n```'))
        self.assertIsNone(repeated_cycle(('short common heading\nvalue 20\nvalue 30\n')*3))

    def test_nonperiodic_duplicate_list_and_distinct_comparison_sections(self):
        values=['apple','pear','orange','lime','melon','lemon','cherry']
        items='\n'.join('- '+v for v in values)
        self.assertIsNotNone(repeated_cycle(items+'\n'+'\n'.join('- '+v for v in reversed(values))))
        self.assertIsNone(repeated_cycle('A:\n'+items+'\nB:\n'+items+'\nC:\n'+items))

    def test_directory_presentation_uses_ids_not_model_merge_claims(self):
        rows=[{'directory_collection':True,'directory_members':[
            {'id':'A','name':'Example Alpha'},{'id':'B','name':'Example Beta'}]}]
        reply='Intro\n- Example Alpha\n- Example Beta\n- Example Alpha (duplicates merged)'
        result,audit=canonical_listing(reply,rows,action='list')
        self.assertEqual(result,'Intro\n- Example Alpha\n- Example Beta')
        self.assertEqual(audit['duplicate_items_removed'],1)
        self.assertEqual(canonical_listing(reply,rows,action='list',aspects=['thickness'])[0],reply)
        self.assertEqual(canonical_listing(reply,rows,action='none')[0],reply)
