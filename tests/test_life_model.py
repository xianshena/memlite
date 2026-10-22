"""GPU-only numerical/causal contracts; CPU model execution is prohibited."""
import unittest

try:
    import torch
    CUDA = torch.cuda.is_available()
except ImportError:
    CUDA = False


@unittest.skipUnless(CUDA, 'requires user CUDA GPU; never substitute CPU execution')
class GPUChoiceTests(unittest.TestCase):
    def setUp(self):
        from life_choice.gpu_model import LargeChoiceConfig, LargeChoiceNetwork
        config = LargeChoiceConfig(width=32, ff_width=64, depth=2, heads=4, dropout=0)
        torch.manual_seed(7)
        self.model = LargeChoiceNetwork(config).cuda().eval()
        # Global context must be identical across candidates, as in real input.
        self.x = torch.randn(3, 12, 40, device='cuda')
        self.x[:, :, :17] = self.x[:, :1, :17]
        self.x[:, :, 35:38] = self.x[:, :1, 35:38]
        self.mask = torch.ones(3, 12, dtype=torch.bool, device='cuda')
        self.mask[:, 3] = False

    def test_candidate_set_permutation_equivariance(self):
        perm = torch.tensor([4, 2, 0, 5, 1, 3, 7, 6, 9, 8, 11, 10], device='cuda')
        with torch.no_grad():
            a = self.model(self.x, self.mask)
            b = self.model(self.x[:, perm], self.mask[:, perm])
        self.assertTrue(torch.allclose(a['utilities'][:, perm], b['utilities'], atol=2e-6))
        self.assertTrue(torch.allclose(a['probabilities'][:, perm], b['probabilities'], atol=2e-6))
        self.assertTrue((a['probabilities'][:, 3] == 0).all())

    def test_masked_tokens_cannot_affect_valid_predictions(self):
        modified = self.x.clone()
        modified[:, 3, 17:35] = 100
        modified[:, 3, 38:40] = 100
        with torch.no_grad():
            a, b = self.model(self.x, self.mask), self.model(modified, self.mask)
        self.assertTrue(torch.allclose(a['utilities'][self.mask], b['utilities'][self.mask], atol=2e-6))

    def test_loss_backward_every_branch_cuda_and_finite(self):
        from life_choice.gpu_model import choice_loss
        self.model.train()
        target_u = torch.randn(3, 12, device='cuda')
        target_o = torch.randn(3, 12, 6, device='cuda')
        loss, parts = choice_loss(self.model(self.x, self.mask), self.mask, target_u, target_o)
        loss.backward()
        self.assertEqual(loss.device.type, 'cuda')
        for name, p in self.model.named_parameters():
            self.assertEqual(p.device.type, 'cuda')
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        self.assertGreater(parts['choice'], 0)

    def test_unknown_ties_have_uniform_target_without_arbitrary_argmax(self):
        from life_choice.gpu_model import teacher_distribution
        target = torch.zeros(2, 12, device='cuda')
        mask = torch.zeros(2, 12, dtype=torch.bool, device='cuda')
        mask[:, [0, 2]] = True
        p = teacher_distribution(target, mask)
        self.assertTrue(torch.equal(p[:, 0], torch.full((2,), .5, device='cuda')))
        self.assertTrue(torch.equal(p[:, 2], torch.full((2,), .5, device='cuda')))
        self.assertEqual(p[:, 1].sum().item(), 0)

    def test_reject_cpu_input_and_all_invalid_mask(self):
        with self.assertRaises(RuntimeError):
            self.model(self.x.cpu(), self.mask.cpu())
        with self.assertRaises(ValueError):
            self.model(self.x, torch.zeros_like(self.mask))
        bad = self.x.clone()
        bad[:, 1, 0] += .1
        with self.assertRaises(ValueError):
            self.model(bad, self.mask)

    def test_bf16_forward_backward_adam_step_changes_parameters(self):
        from life_choice.gpu_model import choice_loss
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, foreach=False)
        before = self.model.utility_head.weight.detach().clone()
        target_u = torch.randn(3, 12, device='cuda')
        target_o = torch.randn(3, 12, 6, device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = self.model(self.x, self.mask)
            loss, _ = choice_loss(prediction, self.mask, target_u, target_o)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, self.model.utility_head.weight))
        self.assertEqual(prediction['probabilities'].dtype, torch.float32)
        self.assertEqual(prediction['probabilities'].device.type, 'cuda')


if __name__ == '__main__':
    unittest.main()
