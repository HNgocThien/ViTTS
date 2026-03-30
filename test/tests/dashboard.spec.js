import { test, expect } from '@playwright/test';

test.describe('TTS Studio Dashboard E2E Tests', () => {

  test.beforeEach(async ({ page }) => {
    // Truy cập phần root url đượng định cấu hình sẵn trong baseURL
    await page.goto('/');
  });

  test('TC01: Giao diện chính tải thành công và hiển thị đủ Menu', async ({ page }) => {
    // Tên tiêu đề cửa sổ web
    await expect(page).toHaveTitle(/TTS Dashboard/i);
    // Tiêu đề App bên trong Sidebar
    const mainTitle = page.locator('h1', { hasText: 'F5-TTS Studio' });
    await expect(mainTitle).toBeVisible();
    
    // Kiểm tra đủ 3 tabs trên thanh điều hướng
    await expect(page.locator('.nav-item').filter({ hasText: '1. Labeling' })).toBeVisible();
    await expect(page.locator('.nav-item').filter({ hasText: '2. Training' })).toBeVisible();
    await expect(page.locator('.nav-item').filter({ hasText: '3. Testing' })).toBeVisible();
  });

  test('TC02: Chuyển đổi giữa các Tabs thành công', async ({ page }) => {
    const labelTab = page.locator('.nav-item').filter({ hasText: '1. Labeling' });
    const trainTab = page.locator('.nav-item').filter({ hasText: '2. Training' });
    const testTab = page.locator('.nav-item').filter({ hasText: '3. Testing' });

    // 1. Labeling Tab kiểm tra nội dung
    await labelTab.click();
    await expect(page.locator('h2', { hasText: 'Data Labeling with MRS' })).toBeVisible();

    // 2. Training Tab kiểm tra nội dung
    await trainTab.click();
    await expect(page.locator('h2', { hasText: 'Train F5-TTS Model' })).toBeVisible();

    // 3. Testing Tab kiểm tra nội dung
    await testTab.click();
    await expect(page.locator('h2', { hasText: 'Generate Audio (Testing)' })).toBeVisible();
  });

  test('TC03: Thẻ Labeling hiển thị giao diện Mimic Recording Studio', async ({ page }) => {
    const labelTab = page.locator('.nav-item').filter({ hasText: '1. Labeling' });
    await labelTab.click();
    
    // Tìm kiếm thẻ iframe chứa Labeling MRS trên cổng 3000
    const iframe = page.locator('iframe[title="Labeling Interface"]');
    await expect(iframe).toBeVisible();
    await expect(iframe).toHaveAttribute('src', 'http://localhost:3000');
  });

  test('TC04 & TC06: Form cấu hình Training khởi tạo với thông số mặc định', async ({ page }) => {
    const trainTab = page.locator('.nav-item').filter({ hasText: '2. Training' });
    await trainTab.click();

    // Base Model default is 'F5TTS-Base'
    const baseModelSelect = page.locator('select').nth(1);
    await expect(baseModelSelect).toHaveValue('F5TTS-Base');

    // Epochs default is 10
    const epochInput = page.locator('input[type="number"]').first();
    await expect(epochInput).toHaveValue('10');

    // Batch Size default is 4
    const batchInput = page.locator('input[type="number"]').nth(1);
    await expect(batchInput).toHaveValue('4');
  });

  test('TC05: Chuỗi thao tác Start Training hiển thị log sinh ra', async ({ page }) => {
    const trainTab = page.locator('.nav-item').filter({ hasText: '2. Training' });
    await trainTab.click();
    
    const startButton = page.locator('button', { hasText: 'Start Training' });
    await expect(startButton).toBeVisible();
    
    // Giả lập Click tiến hành Training
    await startButton.click();
    
    // Quan sát nút bị vô hiệu và đổi sang chế độ Loader (Training in Progress)
    const runningBtn = page.locator('button', { hasText: 'Training in Progress...' });
    await expect(runningBtn).toBeVisible();
    await expect(runningBtn).toBeDisabled();
    
    // Quan sát vùng logs theo dõi quá trình
    const logsContainer = page.locator('.logs-container');
    await expect(logsContainer).toContainText('Training initialized...', { timeout: 3000 });
  });

  test('TC07: Giao diện Testing mặc định có cụm câu mẫu thử', async ({ page }) => {
    const testTab = page.locator('.nav-item').filter({ hasText: '3. Testing' });
    await testTab.click();

    const textarea = page.locator('textarea[placeholder="Type your text here..."]');
    await expect(textarea).toBeVisible();
    await expect(textarea).toHaveValue('Nguyễn Du là một đại thi hào của dân tộc Việt Nam.');
  });

  test('TC08: Validate khoá nút Submit khi xoá văn bản Input', async ({ page }) => {
    const testTab = page.locator('.nav-item').filter({ hasText: '3. Testing' });
    await testTab.click();

    const textarea = page.locator('textarea');
    // Xoá trọn vẹn văn bản
    await textarea.fill('');

    // Nút sinh giọng nói phải bị Disabled
    const generateBtn = page.locator('button', { hasText: 'Generate Voice' });
    await expect(generateBtn).toBeDisabled();
  });

  test('TC09: Sinh Audio từ văn bản thành công (Giả lập)', async ({ page }) => {
    const testTab = page.locator('.nav-item').filter({ hasText: '3. Testing' });
    await testTab.click();

    const generateBtn = page.locator('button', { hasText: 'Generate Voice' });
    
    // Thực hiện gọi hàm Generate
    await generateBtn.click();
    
    // Đảo trạng thái nút trong quá trình chờ HTTP Req call
    const synthBtn = page.locator('button', { hasText: 'Synthesizing...' });
    await expect(synthBtn).toBeVisible();
    await expect(synthBtn).toBeDisabled();

    // Khi backend gửi phản hồi Mock Wav thành công, player audio sẽ được render
    const audioPlayer = page.locator('audio');
    await expect(audioPlayer).toBeVisible({ timeout: 15000 }); // Đợi tối đa 15s cho Inference Return
  });
});
