function generateIdFromHeading(heading) {
  const text = heading.textContent.trim().toLowerCase();
  return text.replace(/[^a-z0-9\s-]/g, '').replace(/\s+/g, '-').replace(/-+/g, '-');
}

function buildTOC() {
  const contentRoot = document.querySelector('.article');
  const tocNav = document.getElementById('toc-nav');
  if (!contentRoot || !tocNav) return;

  const headings = Array.from(contentRoot.querySelectorAll('h2, h3'));
  const ul = document.createElement('ul');
  let currentH2Li = null;
  let currentSubUl = null;

  headings.forEach((h) => {
    if (!h.id) {
      h.id = generateIdFromHeading(h);
    }
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = `#${h.id}`;
    a.textContent = h.textContent;

    if (h.tagName.toLowerCase() === 'h2') {
      currentH2Li = li;
      currentSubUl = document.createElement('ul');
      li.appendChild(a);
      li.appendChild(currentSubUl);
      ul.appendChild(li);
    } else if (h.tagName.toLowerCase() === 'h3') {
      if (!currentH2Li) {
        currentH2Li = document.createElement('li');
        currentSubUl = document.createElement('ul');
        currentH2Li.appendChild(currentSubUl);
        ul.appendChild(currentH2Li);
      }
      const subLi = document.createElement('li');
      subLi.appendChild(a);
      currentSubUl.appendChild(subLi);
    }
  });

  tocNav.innerHTML = '';
  tocNav.appendChild(ul);
}

function setupScrollSpy() {
  const tocNav = document.getElementById('toc-nav');
  if (!tocNav) return;
  const links = Array.from(tocNav.querySelectorAll('a'));
  const map = new Map();

  links.forEach((link) => {
    const id = link.getAttribute('href').slice(1);
    const target = document.getElementById(id);
    if (target) {
      map.set(target, link);
    }
  });

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      const link = map.get(entry.target);
      if (!link) return;
      if (entry.isIntersecting) {
        links.forEach((l) => l.classList.remove('active'));
        link.classList.add('active');
      }
    });
  }, {
    rootMargin: '-30% 0px -60% 0px',
    threshold: [0, 1.0]
  });

  map.forEach((_, section) => observer.observe(section));
}

function setupActions() {
  document.querySelectorAll('[data-action="share"]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const shareData = {
        title: document.title,
        text: 'Check out this article',
        url: location.href
      };
      try {
        if (navigator.share) {
          await navigator.share(shareData);
        } else {
          await navigator.clipboard.writeText(shareData.url);
          alert('Link copied to clipboard');
        }
      } catch {}
    });
  });

  document.querySelectorAll('[data-action="cite"]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const title = document.querySelector('.article-title')?.textContent?.trim() || 'Untitled';
      const year = new Date().getFullYear();
      const citation = `${title}. (${year}). AD Prediction Project.`;
      navigator.clipboard.writeText(citation).then(() => alert('Citation copied to clipboard')).catch(() => alert(citation));
    });
  });
}

window.addEventListener('DOMContentLoaded', () => {
  buildTOC();
  setupScrollSpy();
  setupActions();
});