import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className={clsx('hero__title', styles.heroTitle)}>
          {siteConfig.title}
        </Heading>
        <p className={clsx('hero__subtitle', styles.heroSubtitle)}>
          Portal de apuntes y materiales
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/pia/ut5_ia_aplicada/intro_ia">
            Ir a PIA
          </Link>
        </div>
      </div>
    </header>
  );
}

type SubjectCard = {
  title: string;
  description: string;
  to: string;
  cta: string;
  icon?: ReactNode;
};

function LlmIcon(): ReactNode {
  return (
    <svg
      viewBox="0 0 48 48"
      className={styles.cardIcon}
      role="img"
      aria-label="Icono LLM">
      <rect x="6" y="8" width="36" height="28" rx="8" fill="currentColor" opacity="0.16" />
      <rect x="11" y="13" width="26" height="18" rx="4" fill="none" stroke="currentColor" strokeWidth="2" />
      <circle cx="19" cy="22" r="2" fill="currentColor" />
      <circle cx="29" cy="22" r="2" fill="currentColor" />
      <path d="M19 28c2.5 2 7.5 2 10 0" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <path d="M24 8v-3" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <path d="M16 5h16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

const subjects: SubjectCard[] = [
  {
    title: 'PIA',
    description: 'IA aplicada: Transformers, LLM, Visión artificial e integración.',
    to: '/pia/ut5_ia_aplicada/intro_ia',
    cta: 'Entrar en PIA',
    icon: <LlmIcon />,
  },
  // Por cada materia nueva, añadir un nuevo objeto aquí
];

export default function Home(): ReactNode {
  return (
    <Layout
      title={`Apuntes`}
      description="Apuntes del IES Ágora.">
      <HomepageHeader />
      <main className={styles.mainContent}>
        <div className="container">
          <section className={styles.subjectSection}>
            <Heading as="h2" className={styles.sectionTitle}>
              Módulos profesionales
            </Heading>
            <div className={styles.subjectGrid}>
              {subjects.map((subject) => (
                <article key={subject.title} className={styles.subjectCard}>
                  <div className={styles.cardHead}>
                    <Heading as="h3">{subject.title}</Heading>
                    {subject.icon}
                  </div>
                  <p>{subject.description}</p>
                  <Link className={styles.cardLink} to={subject.to}>
                    {subject.cta}
                  </Link>
                </article>
              ))}
            </div>
          </section>
        </div>
      </main>
    </Layout>
  );
}
